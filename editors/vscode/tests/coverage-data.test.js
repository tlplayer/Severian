'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { buildFileCoverage, parseCoverageMap, parseHits } = require('../coverage-data');

test('preserves u64 region IDs and builds compiler-compatible line coverage', () => {
  const file = '/tmp/coverage source.sev';
  const regions = parseCoverageMap(
    JSON.stringify({
      regions: {
        '18446744073709551614': {
          id: 18446744073709551614,
          function: 'main',
          span: {
            file,
            start: { line: 2, column: 5, byte: 10 },
            end: { line: 2, column: 12, byte: 17 },
          },
          kind: 'Statement',
        },
        '18446744073709551613': {
          id: 18446744073709551613,
          function: 'main',
          span: {
            file,
            start: { line: 2, column: 13, byte: 18 },
            end: { line: 2, column: 20, byte: 25 },
          },
          kind: 'Statement',
        },
        '18446744073709551612': {
          id: 18446744073709551612,
          function: 'main',
          span: {
            file,
            start: { line: 1, column: 1, byte: 0 },
            end: { line: 2, column: 20, byte: 25 },
          },
          kind: 'Function',
        },
      },
    }),
  );
  const hits = parseHits('18446744073709551614\n18446744073709551612\n');
  const coverage = buildFileCoverage(regions, hits).values().next().value;

  assert.deepEqual(
    regions.map((region) => region.id),
    ['18446744073709551614', '18446744073709551613', '18446744073709551612'],
  );
  assert.equal(coverage.lines.get(1).covered, true);
  assert.equal(coverage.lines.get(1).coveredRegions, 1);
  assert.deepEqual(coverage.metrics.lines, { count: 1, covered: 1, percent: 100 });
  assert.deepEqual(coverage.metrics.regions, { count: 2, covered: 1, percent: 50 });
  assert.deepEqual(coverage.metrics.functions, { count: 1, covered: 1, percent: 100 });
});

test('marks an executable line red when none of its statements were hit', () => {
  const regions = [
    {
      id: '42',
      file: '/tmp/uncovered.sev',
      functionName: 'answer',
      kind: 'Statement',
      startLine: 3,
      startColumn: 0,
      endLine: 3,
      endColumn: 6,
    },
  ];
  const coverage = buildFileCoverage(regions, new Set()).values().next().value;

  assert.equal(coverage.lines.get(3).covered, false);
  assert.deepEqual(coverage.metrics.lines, { count: 1, covered: 0, percent: 0 });
});

test('ignores malformed regions and hit lines without losing valid data', () => {
  const regions = parseCoverageMap(
    JSON.stringify({ regions: { bad: {}, good: {
      function: 'valid',
      span: {
        file: '/tmp/valid.sev',
        start: { line: 1, column: 1 },
        end: { line: 1, column: 2 },
      },
      kind: 'Statement',
    } } }),
  );

  assert.equal(regions.length, 1);
  assert.deepEqual([...parseHits('good\nnot-an-id\n7\n')], ['7']);
});
