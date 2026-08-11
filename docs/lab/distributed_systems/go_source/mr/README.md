## Map Reduce
## Assignment

http://nil.csail.mit.edu/6.5840/2025/labs/lab-mr.html

## Components

+--------------------------------------------------+
|                    Coordinator                   |
+--------------------------------------------------+

State Machine

    MAP PHASE
        |
        v
    REDUCE PHASE
        |
        v
       DONE


Responsibilities

1. Map Phase
   a. Create map tasks from input files
   b. Assign map tasks to idle workers
   c. Track task status
      - Idle
      - In Progress
      - Completed
   d. Detect worker timeout/failure
   e. Reassign unfinished map tasks
   f. Wait until all map tasks complete

2. Reduce Phase
   a. Create reduce tasks
   b. Assign reduce tasks to idle workers
   c. Provide intermediate file locations
   d. Track reduce task status
      - Idle
      - In Progress
      - Completed
   e. Detect worker timeout/failure
   f. Reassign unfinished reduce tasks
   g. Wait until all reduce tasks complete

3. Completion
   a. Mark job as finished
   b. Notify workers/job client

+--------------------------------------------------+
|                      Worker                      |
+--------------------------------------------------+

Responsibilities

1. Registration / Polling
   a. Request work from coordinator
   b. Receive task assignment
   c. Report task completion

2. Map Task Execution
   a. Read assigned input file
   b. Execute user-defined Map()
   c. Generate intermediate key/value pairs
   d. Partition data into NReduce buckets
   e. Write intermediate files to local disk
   f. Report intermediate file metadata

3. Reduce Task Execution
   a. Receive reduce task assignment
   b. Read all intermediate files for partition
   c. Sort/group by key
   d. Execute user-defined Reduce()
   e. Generate final output file
   f. Report completion

4. Fault Tolerance
   a. Handle task retries
   b. Re-execute reassigned tasks

## Testing
The testing is primarily handled by CLI which calls the lab with a couple test scenarios like wc.

```bash
# from src/main/ execute
bash test-mr.sh
*** Starting wc test.
--- wc test: PASS
*** Starting indexer test.
--- indexer test: PASS
*** Starting map parallelism test.
--- map parallelism test: PASS
*** Starting reduce parallelism test.
--- reduce parallelism test: PASS
*** Starting job count test.
--- job count test: PASS
*** Starting early exit test.
--- early exit test: PASS
*** Starting crash test.
--- crash test: PASS
*** PASSED ALL TESTS
```

## Appendix

[Google Paper](http://nil.csail.mit.edu/6.5840/2025/papers/mapreduce.pdf)