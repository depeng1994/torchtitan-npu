// Installed only as part of the reviewed control release, never via PR input.
// The deployment must vendor/pin @actions/artifact and its lockfile beforehand.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
async function main() {
  const [name, filename] = process.argv.slice(2);
  if (process.argv.length !== 4 || !/^[a-z0-9][a-z0-9-]{0,99}$/.test(name || ''))
    throw new Error('invalid artifact invocation');
  const absolute = path.resolve(filename);
  const stat = fs.lstatSync(absolute);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1)
    throw new Error('ordinary nonempty artifact file required');
  // Toolkit progress must not corrupt the machine-readable stdout response.
  const output = process.stdout.write.bind(process.stdout);
  process.stdout.write = process.stderr.write.bind(process.stderr);
  const {DefaultArtifactClient} = await import('@actions/artifact');
  const result = await new DefaultArtifactClient().uploadArtifact(
    name, [absolute], path.dirname(absolute), {retentionDays: 7});
  if (!Number.isSafeInteger(result.id) || result.id <= 0)
    throw new Error('missing immutable artifact id');
  output(JSON.stringify({id: result.id}) + '\n');
}
main().catch(() => { process.stderr.write('artifact upload outcome unknown\n'); process.exitCode = 1; });
