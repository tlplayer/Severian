# Problems

This category is a running corpus of small programming problems solved in
Severian. It serves two purposes: exercise the language with recognizable
algorithms, and expose syntax or library friction that is easy to miss in
feature-focused examples.

Every problem should:

- live in one `.sev` file;
- identify the source set and problem in its leading comments;
- use a direct, readable solution rather than hiding the algorithm in a native
  helper;
- include ordinary tests below the solution, covering the representative cases
  and at least one boundary case;
- compile and execute through the normal documentation-example harness.

The planned order is the official [LeetCode 75][leetcode-75] track, followed by
[Top Interview 150][top-150], followed by increasingly difficult problems
selected for new language pressure. The filename prefix records completion
order. Each source begins with
`Track`, `Problem`, `Difficulty`, and comma-separated `Tags` comments. Tags are
many-to-many: a problem may exercise both `dynamic-programming` and `math`
without duplicating its source file.

[leetcode-75]: https://leetcode.com/studyplan/leetcode-75/
[top-150]: https://leetcode.com/studyplan/top-interview-150/

## Current set

| File | Problem | Language pressure exercised |
| --- | --- | --- |
| `01-kids-with-candies.sev` | LeetCode 75: Kids With the Greatest Number of Candies | List construction, two-pass iteration, booleans. |
| `02-find-highest-altitude.sev` | LeetCode 75: Find the Highest Altitude | Accumulators, negative integers, maximum tracking. |
| `03-find-pivot-index.sev` | LeetCode 75: Find Pivot Index | Multiple passes, indexed iteration, early return. |
| `04-increasing-triplet.sev` | LeetCode 75: Increasing Triplet Subsequence | Sentinel-free state, branching, early return. |
| `05-nth-tribonacci.sev` | LeetCode 75: N-th Tribonacci Number | Constant-space DP, recurrence arithmetic, loop initialization. |
| `06-min-cost-climbing-stairs.sev` | LeetCode 75: Min Cost Climbing Stairs | Constant-space DP, indexed access, loop initialization. |
| `07-house-robber.sev` | LeetCode 75: House Robber | Rolling DP state and local maximum selection. |
| `08-unique-paths.sev` | LeetCode 75: Unique Paths | One-dimensional matrix DP and nested loops. |
| `09-can-place-flowers.sev` | LeetCode 75: Can Place Flowers | Greedy mutation and boundary-safe neighbor checks. |
| `10-product-except-self.sev` | LeetCode 75: Product of Array Except Self | Prefix/suffix products and reverse traversal. |
| `11-move-zeroes.sev` | LeetCode 75: Move Zeroes | In-place mutation and write cursors. |
| `12-container-most-water.sev` | LeetCode 75: Container With Most Water | Converging two-pointer search. |
| `13-maximum-average-subarray.sev` | LeetCode 75: Maximum Average Subarray I | Fixed-width sliding window. |
| `14-max-consecutive-ones.sev` | LeetCode 75: Max Consecutive Ones III | Variable-width sliding window. |
| `15-longest-subarray-after-delete.sev` | LeetCode 75: Longest Subarray After Deleting One | Sliding window with a mandatory deletion. |
| `16-find-array-difference.sev` | LeetCode 75: Find the Difference of Two Arrays | Membership and deduplicated list construction. |
| `17-unique-occurrences.sev` | LeetCode 75: Unique Number of Occurrences | Frequency analysis and uniqueness. |
| `18-koko-eating-bananas.sev` | LeetCode 75: Koko Eating Bananas | Binary search over an answer space. |
| `19-successful-pairs.sev` | LeetCode 75: Successful Pairs | Pair counting and threshold arithmetic. |
| `20-domino-tromino-tiling.sev` | LeetCode 75: Domino and Tromino Tiling | Modular recurrence DP. |
| `21-find-peak-element.sev` | LeetCode 75: Find Peak Element | Binary search over local slope. |
| `22-stock-with-transaction-fee.sev` | LeetCode 75: Stock with Transaction Fee | Two-state dynamic programming. |
| `23-counting-bits.sev` | LeetCode 75: Counting Bits | DP over integer halves. |
| `24-single-number.sev` | LeetCode 75: Single Number | Occurrence counting without bitwise syntax. |
| `25-minimum-bit-flips.sev` | LeetCode 75: Minimum Flips for OR | Arithmetic bit decomposition. |
| `26-asteroid-collision.sev` | LeetCode 75: Asteroid Collision | Mutable stack cursor and simulation. |
| `27-daily-temperatures.sev` | LeetCode 75: Daily Temperatures | Forward search and early return. |
| `28-merge-strings-alternately.sev` | LeetCode 75: Merge Strings Alternately | String indexing and concatenation. |
| `29-is-subsequence.sev` | LeetCode 75: Is Subsequence | String indexing and two cursors. |
| `30-maximum-vowels.sev` | LeetCode 75: Maximum Vowels | String sliding window. |
| `31-longest-common-subsequence.sev` | LeetCode 75: Longest Common Subsequence | One-row string DP. |
| `32-edit-distance.sev` | LeetCode 75: Edit Distance | One-row edit DP. |

Progress: **32 / 75** LeetCode 75 problems, **104** attached test cases.

## Syntax observations

Keep observations here even when the current syntax is valid. Repeated friction
is stronger evidence for a language or standard-library change than a single
contrived example.

- Algorithms that need a maximum currently spell out the initial element and a
  loop because there is no general `max(values)` reduction.
- Index-sensitive algorithms use `indices(values)`, which is readable and keeps
  the collection's shape stable while iterating.
- Returning a list still requires a concrete annotation such as `list[bool]`;
  local empty-list element types are inferred from later appends.
- There are no integer infinity literals, so the increasing-triplet solution
  uses explicit `hasFirst` and `hasSecond` state instead of numeric sentinels.
- Loop initialization reads naturally for a single value (`while index <= n
  with index := 3`), while algorithms needing several loop-local cursors still
  initialize the remaining state immediately above the loop.
- Small `minimum` and `maximum` helpers currently read more clearly than
  repeatedly mutating a temporary inside conditionals nested in DP loops.
- Stack algorithms currently manage a logical `top` cursor because lists do not
  yet expose a removal operation; the resulting control flow is noticeably
  noisier than the underlying algorithms.

When a syntax or library improvement lands, update the solution and retain a
short note describing what became simpler.
