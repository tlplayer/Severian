'use strict';

const fs = require('fs');
const path = require('path');

const pathCache = new Map();

function normalizeFilePath(file) {
  const cached = pathCache.get(file);
  if (cached) {
    return cached;
  }

  let normalized = path.resolve(file);
  try {
    normalized = fs.realpathSync.native(normalized);
  } catch {
    // Coverage can include a source that has been moved since the report ran.
  }
  if (process.platform === 'win32') {
    normalized = normalized.toLowerCase();
  }
  pathCache.set(file, normalized);
  return normalized;
}

function parseCoverageMap(contents) {
  const document = JSON.parse(contents);
  if (
    !document ||
    !document.regions ||
    typeof document.regions !== 'object' ||
    Array.isArray(document.regions)
  ) {
    throw new Error('coverage map does not contain a regions object');
  }

  const regions = [];
  for (const [id, region] of Object.entries(document.regions)) {
    const span = region && region.span;
    const start = span && span.start;
    const end = span && span.end;
    if (
      !region ||
      !span ||
      !start ||
      !end ||
      typeof span.file !== 'string' ||
      !Number.isInteger(start.line) ||
      !Number.isInteger(start.column) ||
      !Number.isInteger(end.line) ||
      !Number.isInteger(end.column) ||
      typeof region.kind !== 'string'
    ) {
      continue;
    }

    // Use the object key rather than region.id. Region IDs are u64 values and
    // JSON numbers above 2^53 cannot be represented exactly by JavaScript.
    regions.push({
      id,
      file: normalizeFilePath(span.file),
      functionName: typeof region.function === 'string' ? region.function : '',
      kind: region.kind,
      startLine: start.line - 1,
      startColumn: start.column - 1,
      endLine: end.line - 1,
      endColumn: end.column - 1,
    });
  }
  return regions;
}

function parseHits(contents) {
  return new Set(
    contents
      .split(/\r?\n/u)
      .map((line) => line.trim())
      .filter((line) => /^\d+$/u.test(line)),
  );
}

function metric(ids, hits) {
  let covered = 0;
  for (const id of ids) {
    if (hits.has(id)) {
      covered += 1;
    }
  }
  return { count: ids.size, covered, percent: percent(ids.size, covered) };
}

function percent(count, covered) {
  return count === 0 ? 100 : (covered * 100) / count;
}

function buildFileCoverage(regions, hits) {
  const regionFiles = new Map();
  for (const region of regions) {
    let fileRegions = regionFiles.get(region.file);
    if (!fileRegions) {
      fileRegions = new Map();
      regionFiles.set(region.file, fileRegions);
    }
    fileRegions.set(region.id, region);
  }

  const files = new Map();
  for (const [file, fileRegions] of regionFiles) {
    const statementIds = new Set();
    const branchIds = new Set();
    const functionIds = new Set();
    const lines = new Map();

    for (const region of fileRegions.values()) {
      if (region.kind === 'Statement') {
        statementIds.add(region.id);
        let line = lines.get(region.startLine);
        if (!line) {
          line = { line: region.startLine, ids: new Set(), functions: new Set() };
          lines.set(region.startLine, line);
        }
        line.ids.add(region.id);
        if (region.functionName) {
          line.functions.add(region.functionName);
        }
      } else if (region.kind === 'Branch') {
        branchIds.add(region.id);
      } else if (region.kind === 'Function') {
        functionIds.add(region.id);
      }
    }

    let coveredLines = 0;
    for (const line of lines.values()) {
      line.coveredRegions = [...line.ids].filter((id) => hits.has(id)).length;
      // This matches the compiler report: an executable line is covered when
      // at least one statement region beginning on that line was reached.
      line.covered = line.coveredRegions > 0;
      if (line.covered) {
        coveredLines += 1;
      }
    }

    files.set(file, {
      lines,
      metrics: {
        lines: {
          count: lines.size,
          covered: coveredLines,
          percent: percent(lines.size, coveredLines),
        },
        regions: metric(statementIds, hits),
        branches: metric(branchIds, hits),
        functions: metric(functionIds, hits),
      },
    });
  }
  return files;
}

module.exports = {
  buildFileCoverage,
  normalizeFilePath,
  parseCoverageMap,
  parseHits,
};
