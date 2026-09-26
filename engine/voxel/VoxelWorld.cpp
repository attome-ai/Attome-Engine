#include "VoxelWorld.h"

#include "Lighting.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace atm::voxel {

namespace {

enum class State : uint8_t { Missing, Generating, WaitingRemote, Ready };
enum class JobType : uint8_t { Generate, Mesh };

using Token = std::shared_ptr<std::atomic<bool>>;

inline Token makeToken() { return std::make_shared<std::atomic<bool>>(false); }
inline void cancel(Token &t) {
  if (t) {
    t->store(true, std::memory_order_relaxed);
    t.reset();
  }
}

inline bool inWorldY(int32_t cy) { return cy >= 0 && cy < kWorldChunksY; }

struct Entry {
  ChunkCoord coord;
  std::shared_ptr<const Chunk> chunk;
  State state = State::Missing;
  bool meshDirty = false;
  bool meshedOnce = false;
  bool hasMesh = false;
  bool delivered = false; // a mesh result for this chunk went to the renderer
  uint8_t urgency = 0; // 0 streaming, 1 light-affected neighbour, 2 edit border, 3 edited chunk
  uint64_t genJob = 0, meshJob = 0;
  Token genToken, meshToken;
};

struct Job {
  JobType type = JobType::Generate;
  ChunkCoord coord{};
  uint64_t key = 0;
  uint64_t id = 0;
  uint64_t seq = 0;
  float priority = 0.0f;
  Token token;
  std::array<std::shared_ptr<const Chunk>, 27> hood{}; // mesh jobs
};

struct JobCompare { // max-heap on "less urgent" => top is the smallest priority
  bool operator()(const Job &a, const Job &b) const {
    if (a.priority != b.priority)
      return a.priority > b.priority;
    return a.seq > b.seq;
  }
};

struct Result {
  JobType type = JobType::Generate;
  ChunkCoord coord{};
  uint64_t key = 0;
  uint64_t id = 0;
  bool cancelled = false;
  std::shared_ptr<const Chunk> chunk; // generate
  ChunkMeshData mesh;                 // mesh
  uint32_t revision = 0;
  double micros = 0.0;
};

struct Candidate {
  float priority;
  uint64_t key;
  bool mesh;
};

inline double nowMicros() {
  using namespace std::chrono;
  return double(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()) * 1e-3;
}

} // namespace

struct VoxelWorld::Impl {
  VoxelWorldConfig cfg;
  const BlockRegistry &blocks;
  WorldGenerator gen;

  // --- main-thread state ---
  std::unordered_map<uint64_t, Entry> chunks;
  std::vector<ChunkCoord> foci, lastFoci;
  bool fociChanged = true;
  std::unordered_map<uint64_t, std::shared_ptr<const Chunk>> store; // edited / server-sent data
  std::unordered_set<uint64_t> editedSet;
  std::vector<uint64_t> editedKeys;
  std::unordered_set<uint64_t> expected;
  std::vector<ChunkMeshResult> meshResults;
  std::vector<ChunkCoord> unloaded;
  std::vector<Candidate> candidates;
  std::vector<Result> drained;
  int inFlight = 0;
  uint64_t nextJobId = 1, nextSeq = 0;
  bool needSchedule = true, leftovers = false;
  size_t meshedLastUpdate = 0;
  double avgMesh = 0.0, avgGen = 0.0;
  bool haveMeshSample = false, haveGenSample = false;

  // --- shared with workers ---
  std::mutex qm;
  std::condition_variable qcv;
  std::vector<Job> heap;
  bool stop = false;
  std::mutex rm;
  std::deque<Result> results;
  std::vector<std::thread> workers;

  Impl(const VoxelWorldConfig &c, const BlockRegistry &b) : cfg(c), blocks(b), gen(c.seed) {
    cfg.viewRadiusChunks = std::max(1, cfg.viewRadiusChunks);
    cfg.unloadMarginChunks = std::max(0, cfg.unloadMarginChunks);
    cfg.verticalChunksBelow = std::max(0, cfg.verticalChunksBelow);
    cfg.verticalChunksAbove = std::max(0, cfg.verticalChunksAbove);
    cfg.maxJobsInFlight = std::max(1, cfg.maxJobsInFlight);
    cfg.maxResultsPerUpdate = std::max(1, cfg.maxResultsPerUpdate);
    int n = cfg.workerThreads;
    if (n <= 0) {
      const int hw = int(std::thread::hardware_concurrency());
      n = std::max(1, hw - 2);
    }
    heap.reserve(size_t(cfg.maxJobsInFlight));
    workers.reserve(size_t(n));
    for (int i = 0; i < n; ++i)
      workers.emplace_back([this] { workerMain(); });
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lk(qm);
      stop = true;
      heap.clear();
    }
    qcv.notify_all();
    for (std::thread &t : workers)
      if (t.joinable())
        t.join();
  }

  // ------------------------------------------------------------------ workers
  void workerMain() {
    std::unique_ptr<ChunkLighting::Scratch> lightScratch;
    std::unique_ptr<ChunkMesher::Scratch> meshScratch;
    std::unique_ptr<MeshInput> input;
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(qm);
        qcv.wait(lk, [this] { return stop || !heap.empty(); });
        if (stop)
          return;
        std::pop_heap(heap.begin(), heap.end(), JobCompare{});
        job = std::move(heap.back());
        heap.pop_back();
      }
      Result r;
      r.type = job.type;
      r.coord = job.coord;
      r.key = job.key;
      r.id = job.id;
      if (job.token && job.token->load(std::memory_order_relaxed)) {
        r.cancelled = true;
      } else if (job.type == JobType::Generate) {
        const double t0 = nowMicros();
        auto ch = std::make_shared<Chunk>();
        gen.generate(job.coord, *ch);
        if (cfg.postGenerate)
          cfg.postGenerate(job.coord, *ch);
        r.chunk = std::move(ch);
        r.micros = nowMicros() - t0;
      } else {
        if (!lightScratch) { // lazily, so server workers never allocate these
          lightScratch = ChunkLighting::makeScratch();
          meshScratch = ChunkMesher::makeScratch();
          input = std::make_unique<MeshInput>();
        }
        const double t0 = nowMicros();
        ChunkNeighbourhood hood;
        hood.center = job.coord;
        for (size_t i = 0; i < 27; ++i)
          hood.chunks[i] = job.hood[i].get();
        ChunkLighting::buildMeshInput(hood, blocks, *lightScratch, *input);
        ChunkMesher::mesh(*input, blocks, *meshScratch, r.mesh);
        const Chunk *center = job.hood[size_t(ChunkNeighbourhood::index(0, 0, 0))].get();
        r.revision = center ? center->revision() : 0;
        r.micros = nowMicros() - t0;
      }
      job.hood = {}; // release chunk references before queueing the result
      {
        std::lock_guard<std::mutex> lk(rm);
        results.push_back(std::move(r));
      }
    }
  }

  void submit(Job &&job) {
    job.seq = nextSeq++;
    {
      std::lock_guard<std::mutex> lk(qm);
      heap.push_back(std::move(job));
      std::push_heap(heap.begin(), heap.end(), JobCompare{});
    }
    ++inFlight;
    qcv.notify_one();
  }

  // --------------------------------------------------------------- helpers
  Entry *find(ChunkCoord c) {
    auto it = chunks.find(c.key());
    return it == chunks.end() ? nullptr : &it->second;
  }
  const Entry *find(ChunkCoord c) const {
    auto it = chunks.find(c.key());
    return it == chunks.end() ? nullptr : &it->second;
  }

  float priorityOf(ChunkCoord c, uint8_t urgency) const {
    float best = 1e30f;
    for (const ChunkCoord &f : foci) {
      const float dx = float(c.x - f.x), dy = float(c.y - f.y), dz = float(c.z - f.z);
      best = std::min(best, dx * dx + dy * dy + dz * dz);
    }
    const float dist = foci.empty() ? 1e6f : std::sqrt(best);
    return dist - float(urgency) * 10000.0f;
  }

  void markDirty(Entry &e, uint8_t urgency) {
    e.meshDirty = true;
    e.urgency = std::max(e.urgency, urgency);
    needSchedule = true;
  }

  bool neighboursReady(ChunkCoord c) const {
    for (int dy = -1; dy <= 1; ++dy) {
      if (!inWorldY(c.y + dy))
        continue;
      for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy && !dz)
            continue;
          const Entry *n = find({c.x + dx, c.y + dy, c.z + dz});
          if (!n || n->state != State::Ready)
            return false;
        }
    }
    return true;
  }

  // A chunk's data became available (generated, received or edited).
  void onReady(Entry &e, bool replaced, uint8_t urgency) {
    markDirty(e, urgency);
    if (!replaced)
      return; // neighbours not meshed yet are still dirty and wait for us
    for (int dy = -1; dy <= 1; ++dy)
      for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy && !dz)
            continue;
          Entry *n = find({e.coord.x + dx, e.coord.y + dy, e.coord.z + dz});
          if (n && n->state == State::Ready)
            markDirty(*n, 1);
        }
  }

  void recordEdited(uint64_t key, std::shared_ptr<const Chunk> data) {
    store[key] = std::move(data);
    if (editedSet.insert(key).second)
      editedKeys.push_back(key);
  }

  void emitUnloaded(Entry &e) {
    if (cfg.meshing && e.delivered) {
      ChunkMeshResult r;
      r.coord = e.coord;
      r.revision = e.chunk ? e.chunk->revision() : 0;
      meshResults.push_back(std::move(r)); // empty mesh = drop it
      unloaded.push_back(e.coord);
    }
  }

  // ------------------------------------------------------------- streaming
  bool keep(ChunkCoord c) const {
    const int r = cfg.viewRadiusChunks + cfg.unloadMarginChunks;
    for (const ChunkCoord &f : foci) {
      const int dx = c.x - f.x, dz = c.z - f.z;
      if (dx * dx + dz * dz > r * r)
        continue;
      if (c.y < f.y - cfg.verticalChunksBelow - cfg.unloadMarginChunks ||
          c.y > f.y + cfg.verticalChunksAbove + cfg.unloadMarginChunks)
        continue;
      return true;
    }
    return false;
  }

  void rescan() {
    // Unload chunks beyond radius + margin of every focus.
    for (auto it = chunks.begin(); it != chunks.end();) {
      Entry &e = it->second;
      if (keep(e.coord)) {
        ++it;
        continue;
      }
      cancel(e.genToken);
      cancel(e.meshToken);
      emitUnloaded(e);
      it = chunks.erase(it);
    }
    // Load everything within the view radius of any focus.
    const int r = cfg.viewRadiusChunks;
    for (const ChunkCoord &f : foci) {
      const int y0 = std::max(0, f.y - cfg.verticalChunksBelow);
      const int y1 = std::min(kWorldChunksY - 1, f.y + cfg.verticalChunksAbove);
      for (int dz = -r; dz <= r; ++dz)
        for (int dx = -r; dx <= r; ++dx) {
          if (dx * dx + dz * dz > r * r)
            continue;
          for (int y = y0; y <= y1; ++y) {
            const ChunkCoord c{f.x + dx, y, f.z + dz};
            auto [it, inserted] = chunks.try_emplace(c.key());
            if (!inserted)
              continue;
            Entry &e = it->second;
            e.coord = c;
            if (auto s = store.find(it->first); s != store.end()) {
              e.chunk = s->second;
              e.state = State::Ready;
              // The stored data may have changed while unloaded (server
              // update via setChunkData): loaded neighbours that kept an
              // older mesh built against the previous data must re-mesh.
              onReady(e, true, 0);
            } else if (expected.count(it->first)) {
              e.state = State::WaitingRemote;
            } else {
              e.state = State::Missing;
            }
            needSchedule = true;
          }
        }
    }
  }

  void drainResults() {
    drained.clear();
    {
      std::lock_guard<std::mutex> lk(rm);
      const size_t n = std::min(results.size(), size_t(cfg.maxResultsPerUpdate));
      for (size_t i = 0; i < n; ++i) {
        drained.push_back(std::move(results.front()));
        results.pop_front();
      }
    }
    for (Result &r : drained) {
      --inFlight;
      needSchedule = true;
      if (r.cancelled)
        continue;
      auto it = chunks.find(r.key);
      if (r.type == JobType::Generate) {
        avgGen = haveGenSample ? avgGen * 0.95 + r.micros * 0.05 : r.micros;
        haveGenSample = true;
        if (it == chunks.end())
          continue;
        Entry &e = it->second;
        if (e.state != State::Generating || e.genJob != r.id)
          continue;
        e.genJob = 0;
        e.genToken.reset();
        e.chunk = std::move(r.chunk);
        e.state = State::Ready;
        onReady(e, false, 0);
      } else {
        avgMesh = haveMeshSample ? avgMesh * 0.95 + r.micros * 0.05 : r.micros;
        haveMeshSample = true;
        if (it == chunks.end())
          continue;
        Entry &e = it->second;
        if (e.meshJob != r.id)
          continue; // superseded by a newer mesh job
        e.meshJob = 0;
        e.meshToken.reset();
        e.meshedOnce = true;
        e.hasMesh = !r.mesh.empty();
        e.delivered = true;
        ChunkMeshResult out;
        out.coord = r.coord;
        out.revision = r.revision;
        out.mesh = std::move(r.mesh);
        meshResults.push_back(std::move(out));
        ++meshedLastUpdate;
      }
    }
    drained.clear();
  }

  void schedule() {
    const int budget = cfg.maxJobsInFlight - inFlight;
    if (budget <= 0 || !(needSchedule || leftovers))
      return;
    needSchedule = false;
    candidates.clear();
    for (auto &[key, e] : chunks) {
      if (e.state == State::Missing) {
        candidates.push_back({priorityOf(e.coord, 0), key, false});
      } else if (cfg.meshing && e.state == State::Ready && e.meshDirty) {
        if (e.chunk && e.chunk->isUniform() && blocks.renderMode(e.chunk->uniformBlock()) == BlockRender::None) {
          // Uniform air: nothing to draw, fully open. No job needed.
          e.meshDirty = false;
          e.urgency = 0;
          cancel(e.meshToken);
          e.meshJob = 0;
          if (e.delivered) { // replace the previous mesh with nothing
            ChunkMeshResult r;
            r.coord = e.coord;
            r.revision = e.chunk->revision();
            meshResults.push_back(std::move(r));
          }
          e.hasMesh = false;
          e.meshedOnce = true;
          continue;
        }
        if (neighboursReady(e.coord))
          candidates.push_back({priorityOf(e.coord, e.urgency), key, true});
      }
    }
    const size_t take = std::min(candidates.size(), size_t(budget));
    leftovers = candidates.size() > take;
    std::partial_sort(candidates.begin(), candidates.begin() + ptrdiff_t(take), candidates.end(),
                      [](const Candidate &a, const Candidate &b) { return a.priority < b.priority; });
    for (size_t i = 0; i < take; ++i) {
      const Candidate &cd = candidates[i];
      Entry &e = chunks[cd.key];
      Job job;
      job.coord = e.coord;
      job.key = cd.key;
      job.id = nextJobId++;
      job.priority = cd.priority;
      if (!cd.mesh) {
        job.type = JobType::Generate;
        e.state = State::Generating;
        e.genJob = job.id;
        e.genToken = makeToken();
        job.token = e.genToken;
      } else {
        job.type = JobType::Mesh;
        cancel(e.meshToken); // supersede an older in-flight mesh
        e.meshJob = job.id;
        e.meshToken = makeToken();
        job.token = e.meshToken;
        e.meshDirty = false;
        e.urgency = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz)
            for (int dx = -1; dx <= 1; ++dx) {
              const Entry *n = (dx || dy || dz) ? find({e.coord.x + dx, e.coord.y + dy, e.coord.z + dz}) : &e;
              if (n && n->state == State::Ready)
                job.hood[size_t(ChunkNeighbourhood::index(dx, dy, dz))] = n->chunk;
            }
      }
      submit(std::move(job));
    }
  }
};

// =========================================================================

VoxelWorld::VoxelWorld(const VoxelWorldConfig &config, const BlockRegistry &blocks)
    : impl_(std::make_unique<Impl>(config, blocks)), blocks_(blocks) {}

VoxelWorld::~VoxelWorld() = default;

void VoxelWorld::clearFoci() { impl_->foci.clear(); }

void VoxelWorld::setViewRadius(int chunks) {
  chunks = std::clamp(chunks, 1, 32);
  if (chunks == impl_->cfg.viewRadiusChunks) return;
  impl_->cfg.viewRadiusChunks = chunks;
  impl_->lastFoci.clear(); // force a rescan
}

int VoxelWorld::viewRadius() const { return impl_->cfg.viewRadiusChunks; }

void VoxelWorld::addFocus(double x, double y, double z) {
  const auto cc = [](double v) {
    const double f = std::floor(v / double(kChunkSize));
    return int32_t(std::clamp(f, -33554432.0, 33554431.0));
  };
  impl_->foci.push_back({cc(x), cc(y), cc(z)});
}

void VoxelWorld::update() {
  Impl &m = *impl_;
  m.meshedLastUpdate = 0;
  m.drainResults();
  if (m.foci != m.lastFoci) {
    m.lastFoci = m.foci;
    m.rescan();
  }
  m.schedule();
}

void VoxelWorld::takeMeshResults(std::vector<ChunkMeshResult> &out) {
  auto &src = impl_->meshResults;
  if (out.empty()) {
    out.swap(src);
  } else {
    for (ChunkMeshResult &r : src)
      out.push_back(std::move(r));
  }
  src.clear();
}

void VoxelWorld::takeUnloaded(std::vector<ChunkCoord> &out) {
  auto &src = impl_->unloaded;
  out.insert(out.end(), src.begin(), src.end());
  src.clear();
}

BlockId VoxelWorld::blockAt(BlockPos p) const {
  if (p.y < 0 || p.y >= kWorldHeight)
    return kAir;
  const Entry *e = impl_->find(chunkOf(p));
  if (!e || e->state != State::Ready || !e->chunk)
    return kAir;
  const LocalPos l = localOf(p);
  return e->chunk->get(l.x, l.y, l.z);
}

bool VoxelWorld::isLoaded(ChunkCoord c) const {
  if (!inWorldY(c.y))
    return true; // nothing there to load: open sky / below the world
  const Entry *e = impl_->find(c);
  return e && e->state == State::Ready;
}

bool VoxelWorld::setBlock(BlockPos p, BlockId id) {
  if (p.y < 0 || p.y >= kWorldHeight)
    return false;
  Impl &m = *impl_;
  const ChunkCoord c = chunkOf(p);
  Entry *e = m.find(c);
  if (!e || e->state != State::Ready || !e->chunk)
    return false;
  const LocalPos l = localOf(p);
  const BlockId old = e->chunk->get(l.x, l.y, l.z);
  if (old == id)
    return true;
  auto copy = std::make_shared<Chunk>(*e->chunk); // copy-on-write
  copy->set(l.x, l.y, l.z, id);
  e->chunk = copy;
  m.recordEdited(c.key(), e->chunk);
  m.markDirty(*e, 3);

  // Neighbours: AO/culling at borders (urgent), light within 15 blocks.
  const bool lightChanged = m.blocks.transmitsLight(old) != m.blocks.transmitsLight(id) ||
                            m.blocks.emission(old) != m.blocks.emission(id);
  const int lc[3] = {l.x, l.y, l.z};
  for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz)
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy && !dz)
          continue;
        const int d[3] = {dx, dy, dz};
        int maxDist = 0, sumDist = 0;
        for (int a = 0; a < 3; ++a) {
          const int dist = d[a] < 0 ? lc[a] + 1 : d[a] > 0 ? kChunkSize - lc[a] : 0;
          maxDist = std::max(maxDist, dist);
          sumDist += dist;
        }
        uint8_t urgency;
        if (maxDist <= 1)
          urgency = 2;
        else if (lightChanged && sumDist <= ChunkLighting::kMargin)
          urgency = 1;
        else
          continue;
        Entry *n = m.find({c.x + dx, c.y + dy, c.z + dz});
        if (n && n->state == State::Ready)
          m.markDirty(*n, urgency);
      }
  return true;
}

void VoxelWorld::setChunkData(ChunkCoord c, std::shared_ptr<const Chunk> data) {
  if (!data || !inWorldY(c.y))
    return;
  Impl &m = *impl_;
  const uint64_t key = c.key();
  m.recordEdited(key, data);
  m.expected.erase(key);
  Entry *e = m.find(c);
  if (!e)
    return; // kept in the store; used when the chunk streams in
  const bool replaced = e->state == State::Ready;
  cancel(e->genToken);
  e->genJob = 0;
  e->chunk = std::move(data);
  e->state = State::Ready;
  m.onReady(*e, replaced, replaced ? 3 : 0);
}

std::shared_ptr<const Chunk> VoxelWorld::chunk(ChunkCoord c) const {
  const Entry *e = impl_->find(c);
  return (e && e->state == State::Ready) ? e->chunk : nullptr;
}

const std::vector<uint64_t> &VoxelWorld::editedChunkKeys() const { return impl_->editedKeys; }

bool VoxelWorld::isEdited(ChunkCoord c) const { return impl_->editedSet.count(c.key()) != 0; }

void VoxelWorld::setExpectedEdited(std::span<const uint64_t> keys) {
  Impl &m = *impl_;
  for (uint64_t key : keys) {
    if (m.store.count(key))
      continue; // already have the authoritative data
    m.expected.insert(key);
    auto it = m.chunks.find(key);
    if (it == m.chunks.end())
      continue;
    Entry &e = it->second;
    if (e.state == State::Missing || e.state == State::Generating) {
      cancel(e.genToken);
      e.genJob = 0;
      e.state = State::WaitingRemote;
    }
  }
}

const WorldGenerator &VoxelWorld::generator() const { return impl_->gen; }

VoxelWorld::Stats VoxelWorld::stats() const {
  const Impl &m = *impl_;
  Stats s;
  for (const auto &[key, e] : m.chunks) {
    if (e.state == State::Ready && e.chunk) {
      ++s.loadedChunks;
      s.memoryBytes += e.chunk->memoryBytes();
    }
  }
  s.pendingJobs = size_t(std::max(0, m.inFlight));
  s.meshedLastUpdate = m.meshedLastUpdate;
  s.avgMeshMicros = m.avgMesh;
  s.avgGenMicros = m.avgGen;
  return s;
}

} // namespace atm::voxel
