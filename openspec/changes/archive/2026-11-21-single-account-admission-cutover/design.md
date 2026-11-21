# Design

## Decisions

### 1. Re-check the locked file

An account can replace its own lock file while a load waits, for example by cleaning its state directory. The waiter would then hold the old file while a new load holds the new one, and both would pass the memory guard. So after `flock`, the path must still name the locked file (device and inode). If it doesn't, the load locks the new file. The retry counts as contention, so it runs under the same deadline.

### 2. The deadline applies to acquisitions

Polling every 20 ms means a holder can release between the deadline and the next try. Once the lock has been found held, an acquisition at or after the deadline is a timeout. An uncontended first attempt still succeeds immediately, whatever the wait.
