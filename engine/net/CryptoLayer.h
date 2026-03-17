#pragma once

#include "ConnectionSlots.h"
#include "NetTypes.h"

#include <cstdint>

namespace attome::net {

#if NET_ENCRYPTION

class CryptoLayer {
public:
  bool init();

  uint32_t generate_conn_id() const;

  void set_psk(ConnectionSlots &slots, ConnId conn) const;

#if NET_ENCRYPTION_KEY_MODE == 1
  const uint8_t *server_public_key() const { return server_pk_; }

  bool server_derive_key(ConnectionSlots &slots, ConnId conn,
                         const uint8_t client_pk[32]) const;
  bool client_derive_key(uint8_t out_key[32], const uint8_t client_sk[32],
                         const uint8_t server_pk[32],
                         uint32_t conn_id) const;
  bool generate_keypair(uint8_t out_pk[32], uint8_t out_sk[32]) const;
#endif

  bool encrypt_in_place(uint8_t *buf, uint16_t &len, ConnId conn,
                        const ConnectionSlots &slots) const;
  bool decrypt_in_place(uint8_t *buf, uint16_t &len, ConnId conn,
                        const ConnectionSlots &slots) const;

private:
#if NET_ENCRYPTION_KEY_MODE == 1
  uint8_t server_pk_[32]{};
  uint8_t server_sk_[32]{};
#endif
};

#else

class CryptoLayer {
public:
  bool init() { return true; }
  uint32_t generate_conn_id() const { return 0; }
  void set_psk(ConnectionSlots &, ConnId) const {}
  bool encrypt_in_place(uint8_t *, uint16_t &, ConnId,
                        const ConnectionSlots &) const {
    return true;
  }
  bool decrypt_in_place(uint8_t *, uint16_t &, ConnId,
                        const ConnectionSlots &) const {
    return true;
  }
};

#endif

} // namespace attome::net

