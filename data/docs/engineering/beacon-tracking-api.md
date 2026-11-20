# Beacon Tracking API v2

Base URL: https://api.brackenfield.example/beacon/v2

## Authentication

Each request needs an API key in the `X-Beacon-Key` header. Keys are created in the business account portal. Each account can have up to 5 active keys.

## Rate limits

- 60 requests per minute per API key.
- Bulk endpoint: up to 100 tracking numbers per request.
- Exceeding the limit returns HTTP 429 with a `Retry-After` header in seconds.

## Endpoints

- `GET /parcels/{tracking_number}` — current status and event history.
- `POST /parcels/bulk` — status for up to 100 parcels.
- `POST /webhooks` — register a URL for status-change events.

## Tracking statuses

CREATED, COLLECTED, AT_DEPOT, OUT_FOR_DELIVERY, DELIVERED, ATTEMPTED, HELD, RETURNING, RETURNED, EXCEPTION.

EXCEPTION means the parcel needs manual handling, for example a damaged label or a customs hold.

## Webhooks

- Webhooks are sent as HTTP POST with a JSON body.
- Each event has an `event_id`. Since INC-2025-009, the same status change always carries the same `event_id`, so receivers should de-duplicate on it.
- Failed webhooks are retried 6 times over 2 hours with exponential backoff.
- Each webhook is signed with HMAC-SHA256 in the `X-Beacon-Signature` header.

## Differences from v1

- v2 added webhooks; v1 was polling only.
- v2 renamed the status IN_TRANSIT to AT_DEPOT.
- v1 allowed 30 requests per minute; v2 allows 60.
- v1 was switched off on 31 January 2025.

## Support

Email developers@brackenfield.example. The Platform team answers within 1 business day.
