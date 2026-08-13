# CuKD FG-DS Wi-Fi UDP Protocol v2

Protocol identifier: `cukd_fgds_wifi_udp_session_v2`.

## Network contract

- IPv4 UDP in Wi-Fi station mode.
- Device UDP port: `42101`.
- Host UDP port: `42102`.
- Maximum application datagram: 768 bytes.
- One outstanding stop-and-wait transaction.
- Maximum encoded attempt number: 255.
- The device binds the first valid session datagram source IPv4 address and
  source port. Other endpoints are ignored for that provisioned session.

## Outer envelope

Every request and response is printable ASCII with seven comma-separated
fields. `INNER_HEX` is uppercase hexadecimal encoding of nonempty printable
ASCII. `CRC16` is four uppercase hexadecimal digits containing
CRC-16-CCITT-FALSE over all preceding comma-separated fields.

Request:

```text
CUKDW2Q,SESSION32,STAGE16,TX16,ATTEMPT,INNER_HEX,CRC16
```

Response:

```text
CUKDW2R,SESSION32,STAGE16,TX16,ATTEMPT,INNER_HEX,CRC16
```

`SESSION32` is 32 uppercase hexadecimal digits. `STAGE16` and `TX16` are
16 uppercase hexadecimal digits. Decimal fields are canonical: no sign and no
leading zero except the value zero where zero is permitted.

The host accepts a response only when source endpoint, envelope direction,
session, stage, transaction, attempt, payload encoding, and CRC all match.
Ignored datagrams are counted by rejection category.

## Runtime provisioning

The firmware contains no configured SSID or password. USB serial provisioning
uses this line:

```text
CUKDWCFG2,SESSION32,42101,SSID_HEX,PASSWORD_HEX,CRC16
```

SSID is 1 to 32 printable ASCII bytes. Password is 8 to 63 printable ASCII
bytes. The response is:

```text
CUKDWCFG2R,SESSION32,STATUS,IPV4,PORT,RSSI_DBM,FIRMWARE_HEX,MAC,CRC16
```

On success, `STATUS` is `OK`, `PORT` is 42101, and `IPV4` is the device's
unicast address. The host closes serial before opening the UDP replay socket.

## Build identity

The serial and UDP identity query inner payload is:

```text
CUKDWID?
```

The exact response is:

```text
CUKDWBUILD,student,export_id,wireless_bundle_id,board,cukd_fgds_wifi_udp_session_v2
```

The export and bundle identifiers are 64-character SHA-256-derived values. The
host requires exact identity before every stage. This binds experiment evidence
to the compiled source bundle but is not secure-boot attestation.

## Stage controls

Reserved transaction IDs:

| Control | TX16 |
|---|---|
| Identity | `FFFFFFFFFFFFFFFC` |
| Abort | `FFFFFFFFFFFFFFFD` |
| Begin | `FFFFFFFFFFFFFFFE` |
| End | `FFFFFFFFFFFFFFFF` |

Begin request and acknowledgement:

```text
CUKDWBEGIN,STAGE16,ORDINAL,EXPECTED_ROWS
CUKDWBEGINR,STAGE16,ORDINAL,EXPECTED_ROWS,OK
```

End request and acknowledgement:

```text
CUKDWEND,STAGE16,COMPLETED_ROWS
CUKDWENDR,STAGE16,COMPLETED_ROWS,EXPECTED_ROWS,RECEIVED_DATAGRAMS,
OVERSIZED_DATAGRAMS,SHORT_READS,BAD_ENVELOPES,WRONG_SESSIONS,
WRONG_ENDPOINTS,WRONG_STAGES,CONTROL_ERRORS,DATA_ERRORS,
DUPLICATE_REPLAYS,STALE_TRANSACTIONS,INFERENCES,ORDINAL,OK
```

The displayed end response is one physical line; it is wrapped above only for
readability. A stage passes only when completed rows, expected rows, and device
inferences all equal the frozen stage size.

Abort request and acknowledgement:

```text
CUKDWABORT,STAGE16,COMPLETED_ROWS
CUKDWABORTR,STAGE16,COMPLETED_ROWS,EXPECTED_ROWS,INFERENCES,OK
```

Ordinals are monotonic within a provisioned session. Reusing or decreasing an
ordinal is rejected. Stage IDs are random 64-bit nonces and must differ across
the three required stages.

## Inference payload

The inner data request reuses the strict `CUKD1` protocol:

```text
CUKD1,row_id,f0,...,f16,CRC16
```

The inner response is:

```text
CUKD1R,row_id,status,pred,l0,...,l4,preprocess_us,inference_us,total_us,CRC16
```

Data transaction ID is the row ID encoded as 16 uppercase hexadecimal digits.
Rows must be ordered from zero without gaps. The inner CRC uses the same
CRC-16-CCITT-FALSE definition as the outer envelope.

## Retry and idempotence rules

The host retransmits only after a response timeout. The firmware caches the
latest accepted row request, logits, prediction, and compute timings. A retry of
that exact transaction returns the cached response and does not run inference a
second time. A reused transaction ID with different content is rejected.

Begin is idempotent for an identical active-stage contract. End responses are
cached so a lost end acknowledgement can be replayed after the stage closes.
Abort responses are also cached until a later stage begins. Because
stop-and-wait permits only one outstanding row, abort recovery checks the two
possible device counts when the final data response is uncertain.

`response_timeouts` is not measured packet loss. A timeout cannot distinguish a
lost request, a lost response, device delay, host scheduling delay, or network
delay.

## Timing boundaries

- `preprocess_us`: MCU `micros()` interval around integer standardization.
- `inference_us`: MCU `micros()` interval around fixed-point forward inference.
- `total_us`: sum of the two MCU intervals. UDP parsing, response formatting,
  and transmission are outside the timed code region. The `micros()` interval
  is wall-clock time and can include interrupt preemption.
- `host_observed_datagram_rtt_us`: host monotonic interval from immediately
  before the successful attempt's `sendto` through the matching `recvfrom`.
- `transaction_elapsed_us`: host monotonic interval from the first attempt's
  `sendto` through the matching response, including prior timeouts and retries.

The host and MCU clocks are independent. Subtracting `total_us` from host RTT is
retained only as a descriptive difference and is not pure wireless latency.
