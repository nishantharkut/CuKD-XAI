# CuKD-XAI Serial Replay Protocol

The protocol is line-oriented ASCII to make long-running logs auditable.

## Request

Format:

```text
CUKD1,row_id,f0,f1,...,f16,CRC16\n
```

Fields:

- `CUKD1`: start marker.
- `row_id`: non-negative integer row identifier.
- `f0..f16`: exactly 17 fixed-point WSN-DS feature values.
- `CRC16`: uppercase hexadecimal CRC-16-CCITT-FALSE over the comma-separated
  request body before the CRC field.

The 17 feature values are already extracted WSN-DS numeric features encoded as
fixed-point integers. They are not raw radio packets or raw network frames.

## Response

Format:

```text
CUKD1R,row_id,status,predicted_class,l0,l1,l2,l3,l4,preprocess_us,inference_us,total_us,CRC16\n
```

Fields:

- `CUKD1R`: response marker.
- `row_id`: row identifier copied from the request.
- `status`: one of the status values below.
- `predicted_class`: predicted class index, or `-1` for invalid requests.
- `l0..l4`: fixed-point output logits or zeros for invalid requests.
- `preprocess_us`: device-side preprocessing time.
- `inference_us`: device-side inference time.
- `total_us`: total request service time on the device.
- `CRC16`: CRC-16-CCITT-FALSE over the response body before the CRC field.

## Status Values

- `OK`
- `BAD_START`
- `BAD_LENGTH`
- `BAD_CHECKSUM`
- `BAD_FEATURE_RANGE`
- `INTERNAL_ERROR`

The host must treat timeouts, malformed responses, duplicate row IDs, missing
row IDs, and out-of-order responses as run failures unless explicitly
documented.

