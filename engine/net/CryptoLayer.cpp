#include "CryptoLayer.h"

#if NET_ENCRYPTION

#include <sodium.h>

#include <array>
#include <cstring>

namespace attome::net {

static constexpr std::array<uint8_t, 32> kPsk = NET_PSK;

static inline void write_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
  dst[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
  dst[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

static inline void write_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
}

static inline void derive_nonce(uint8_t out_nonce[12], uint32_t conn_id,
                                uint8_t channel, uint16_t seq) {
  write_u32_le(&out_nonce[0], conn_id);
  out_nonce[4] = channel;
  write_u16_le(&out_nonce[5], seq);
  std::memset(&out_nonce[7], 0, 5);
}

bool CryptoLayer::init() {
  if (sodium_init() < 0) {
    return false;
  }

#if NET_ENCRYPTION_KEY_MODE == 1
  crypto_box_keypair(server_pk_, server_sk_);
#endif

  return true;
}

uint32_t CryptoLayer::generate_conn_id() const {
  uint32_t v = randombytes_random();
  if (v == 0) {
    v = 1;
  }
  return v;
}

void CryptoLayer::set_psk(ConnectionSlots &slots, ConnId conn) const {
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return;
  }
  std::memcpy(slots.encryption_key[conn], kPsk.data(), kPsk.size());
}

#if NET_ENCRYPTION_KEY_MODE == 1

bool CryptoLayer::generate_keypair(uint8_t out_pk[32],
                                   uint8_t out_sk[32]) const {
  if (out_pk == nullptr || out_sk == nullptr) {
    return false;
  }
  crypto_box_keypair(out_pk, out_sk);
  return true;
}

static inline bool kdf_expand(uint8_t out_key[32], const uint8_t master[32],
                              uint32_t conn_id) {
  static_assert(crypto_kdf_CONTEXTBYTES == 8, "Unexpected KDF context size.");
  const char ctx[crypto_kdf_CONTEXTBYTES] = {'A', 'T', 'M', 'N',
                                             'E', 'T', '0', '1'};
  const uint64_t subkey_id = static_cast<uint64_t>(conn_id);
  return crypto_kdf_derive_from_key(out_key, 32, subkey_id, ctx, master) == 0;
}

bool CryptoLayer::server_derive_key(ConnectionSlots &slots, ConnId conn,
                                    const uint8_t client_pk[32]) const {
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return false;
  }
  if (client_pk == nullptr) {
    return false;
  }

  uint8_t shared[crypto_scalarmult_BYTES]{};
  if (crypto_scalarmult(shared, server_sk_, client_pk) != 0) {
    return false;
  }

  const uint32_t cid = slots.encryption_conn_id[conn];
  return kdf_expand(slots.encryption_key[conn], shared, cid);
}

bool CryptoLayer::client_derive_key(uint8_t out_key[32],
                                    const uint8_t client_sk[32],
                                    const uint8_t server_pk[32],
                                    uint32_t conn_id) const {
  if (out_key == nullptr || client_sk == nullptr || server_pk == nullptr) {
    return false;
  }

  uint8_t shared[crypto_scalarmult_BYTES]{};
  if (crypto_scalarmult(shared, client_sk, server_pk) != 0) {
    return false;
  }
  return kdf_expand(out_key, shared, conn_id);
}

#endif

bool CryptoLayer::encrypt_in_place(uint8_t *buf, uint16_t &len, ConnId conn,
                                  const ConnectionSlots &slots) const {
  if (buf == nullptr) {
    return false;
  }
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return false;
  }
  if (len < NET_HEADER_SIZE) {
    return false;
  }

  uint8_t nonce[crypto_aead_chacha20poly1305_IETF_NPUBBYTES]{};
  const auto *hdr = reinterpret_cast<const PacketHeader *>(buf);
  derive_nonce(nonce, slots.encryption_conn_id[conn], hdr->channel, hdr->seq);

  unsigned long long maclen = 0;
  uint8_t *c = buf + NET_HEADER_SIZE;
  const unsigned long long mlen =
      static_cast<unsigned long long>(len - NET_HEADER_SIZE);
  uint8_t *mac = buf + len;

  if (crypto_aead_chacha20poly1305_ietf_encrypt_detached(
          c, mac, &maclen, c, mlen, buf, NET_HEADER_SIZE, nullptr, nonce,
          slots.encryption_key[conn]) != 0) {
    return false;
  }

  if (maclen != NET_ENCRYPTION_TAG_SIZE) {
    return false;
  }

  len = static_cast<uint16_t>(len + NET_ENCRYPTION_TAG_SIZE);
  return true;
}

bool CryptoLayer::decrypt_in_place(uint8_t *buf, uint16_t &len, ConnId conn,
                                  const ConnectionSlots &slots) const {
  if (buf == nullptr) {
    return false;
  }
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return false;
  }
  if (len < (NET_HEADER_SIZE + NET_ENCRYPTION_TAG_SIZE)) {
    return false;
  }

  uint8_t nonce[crypto_aead_chacha20poly1305_IETF_NPUBBYTES]{};
  const auto *hdr = reinterpret_cast<const PacketHeader *>(buf);
  derive_nonce(nonce, slots.encryption_conn_id[conn], hdr->channel, hdr->seq);

  uint8_t *c = buf + NET_HEADER_SIZE;
  const unsigned long long clen =
      static_cast<unsigned long long>(len - NET_HEADER_SIZE -
                                      NET_ENCRYPTION_TAG_SIZE);
  const uint8_t *mac = buf + (len - NET_ENCRYPTION_TAG_SIZE);

  if (crypto_aead_chacha20poly1305_ietf_decrypt_detached(
          c, nullptr, c, clen, mac, buf, NET_HEADER_SIZE, nonce,
          slots.encryption_key[conn]) != 0) {
    return false;
  }

  len = static_cast<uint16_t>(len - NET_ENCRYPTION_TAG_SIZE);
  return true;
}

} // namespace attome::net

#endif

