# ADR-007: Nightly batch plan plus intraday re-plans in Sorter

- Status: Accepted
- Date: 2024-02-05
- Deciders: Routing team, Ines Corrado (COO)

## Context

Until 2024, Sorter computed all routes once, in a nightly batch, and routes did not change during the day. Same Day bookings (cut-off 11:00) had to be added by depot staff by hand, and traffic problems could not be routed around. In 2023, 6% of Same Day parcels missed their delivery window.

## Decision

Keep the nightly batch plan at 02:00, and add intraday re-plans every 30 minutes from 07:00 to 19:00. A re-plan may only change stops a driver has not yet reached, and may move at most 8 stops per van per re-plan, so drivers are not confused by large changes.

## Reasons

- New Same Day bookings are added to routes automatically within 30 minutes.
- Live traffic data can be used.
- Limiting changes to 8 stops per van keeps routes predictable for drivers.

## Rejected options

- Fully continuous re-planning: rejected because routes changed too often in a trial and drivers missed stops.
- Hourly re-plans: rejected because Same Day parcels booked just before 11:00 waited too long.

## Consequences

- Sorter's database load rose about 4 times during the day.
- Tally must show route changes to the driver with a notification.
- Result: in the first six months, Same Day parcels missing their window fell from 6% to 2.5%.
