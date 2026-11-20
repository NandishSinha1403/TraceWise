# INC-2025-009: Duplicate DELIVERED webhook events

- Severity: SEV2
- Date: 3 June 2025
- Duration: 2 hours 40 minutes
- Incident commander: Platform team engineering manager

## Summary

Beacon sent duplicate DELIVERED webhook events for about 12,000 parcels. Several business customers sent their own customers two "your parcel has arrived" emails, and one customer's warehouse system closed orders twice.

## Root cause

A slow downstream queue caused Beacon's event publisher to time out and retry. Each retry generated a new `event_id`, so receivers could not tell retries from new events.

## Fix

- Each status change now has a deterministic `event_id`, built from the tracking number, the status, and the scan time. Retries reuse it.
- The Beacon documentation now tells receivers to de-duplicate on `event_id`.

## Follow-up actions

1. Deterministic event IDs. Owner: Platform team. Done June 2025.
2. Load test the event publisher at 3 times peak volume before the November peak freeze. Owner: Platform team. Done October 2025.
3. Email all webhook customers explaining the change. Done June 2025.
