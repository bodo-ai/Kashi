const {test} = require('node:test');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const workflow = fs.readFileSync(path.join(__dirname, '../workflows/pr-review.yml'), 'utf8');

function block(stepName, key, index = 1) {
  const step = workflow.split(`      - name: ${stepName}\n`)[index].split('\n      - ')[0];
  const marker = key === 'run' ? '        run: |\n' : '          script: |\n';
  const lines = step.split(marker)[1].split('\n');
  const indent = key === 'run' ? 10 : 12;
  return lines.filter(line => line.startsWith(' '.repeat(indent))).map(line => line.slice(indent)).join('\n');
}

function git(cwd, ...args) {
  return execFileSync('git', args, {cwd, encoding: 'utf8'}).trim();
}

test('PR review excludes base-only changes and retains changed spec fingerprints', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'spec-review-'));
  try {
    git(root, 'init', '-q');
    git(root, 'config', 'user.email', 'test@example.com');
    git(root, 'config', 'user.name', 'Test');
    fs.mkdirSync(path.join(root, 'specs'));
    fs.writeFileSync(path.join(root, 'specs/changed.md'), 'Original\n');
    fs.writeFileSync(path.join(root, 'specs/base-only.md'), 'Original\n');
    fs.writeFileSync(path.join(root, 'base-only.txt'), 'Original\n');
    git(root, 'add', '.');
    git(root, 'commit', '-qm', 'common ancestor');
    const ancestor = git(root, 'rev-parse', 'HEAD');
    fs.writeFileSync(path.join(root, 'specs/base-only.md'), 'Base-only change\n');
    fs.writeFileSync(path.join(root, 'base-only.txt'), 'Base-only change\n');
    git(root, 'commit', '-qam', 'base advances');
    const base = git(root, 'rev-parse', 'HEAD');
    git(root, 'checkout', '-q', '--detach', ancestor);
    fs.writeFileSync(path.join(root, 'specs/changed.md'), 'PR change\n');
    git(root, 'commit', '-qam', 'PR change');
    const head = git(root, 'rev-parse', 'HEAD');
    const output = path.join(root, 'outputs');
    const env = {...process.env, BASE_SHA: base, HEAD_SHA: head, RUNNER_TEMP: root, GITHUB_OUTPUT: output};
    execFileSync('bash', ['-e', '-c', block('Fingerprint specification diff', 'run')], {cwd: root, env});
    const specDiff = fs.readFileSync(path.join(root, 'spec.diff'), 'utf8');
    assert.match(specDiff, /specs\/changed\.md/);
    assert.doesNotMatch(specDiff, /specs\/base-only\.md/);
    assert.match(fs.readFileSync(output, 'utf8'), /changed=true/);
    assert.match(fs.readFileSync(output, 'utf8'), /other_changed=false/);

    const outputs = {};
    const script = block('Check previous reviews', 'script');
    const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
    const oldCwd = process.cwd();
    const oldBase = process.env.BASE_SHA;
    const oldHead = process.env.HEAD_SHA;
    try {
      process.chdir(root);
      process.env.BASE_SHA = base;
      process.env.HEAD_SHA = head;
      const context = {repo: {owner: 'owner', repo: 'repo'}, issue: {number: 1}};
      const core = {setOutput: (key, value) => { outputs[key] = value; }};
      const run = new AsyncFunction('require', 'github', 'context', 'core', script);
      const github = {paginate: async () => [], rest: {issues: {listComments() {}}}};
      await run(require, github, context, core);
      assert.deepEqual(Object.keys(JSON.parse(outputs.pending)), ['specs/changed.md']);
      assert.equal(outputs.review, true);

      const pending = outputs.pending;
      const record = Buffer.from(pending).toString('base64');
      github.paginate = async () => [{user: {login: 'github-actions[bot]'}, body:
        `<!-- agentic-review:spec-record -->\n<!-- spec-reviewed:${record} -->`}];
      await run(require, github, context, core);
      assert.deepEqual(JSON.parse(outputs.pending), {});
      assert.equal(outputs.review, false);

      execFileSync('bash', ['-e', '-c', block('Prepare review input', 'run')], {
        cwd: root, env: {...env, PENDING_SPECS: pending},
      });
      const specInput = fs.readFileSync(path.join(root, 'review-input.md'), 'utf8');
      assert.match(specInput, /specs\/changed\.md/);
      assert.doesNotMatch(specInput, /specs\/base-only\.md/);

      execFileSync('bash', ['-e', '-c', block('Prepare review input', 'run', 2)], {
        cwd: root, env: {...env, PR_BODY: ''},
      });
      const prInput = fs.readFileSync(path.join(root, 'review-input.md'), 'utf8');
      assert.match(prInput, /specs\/changed\.md/);
      assert.doesNotMatch(prInput, /base-only/);
    } finally {
      process.chdir(oldCwd);
      if (oldBase === undefined) delete process.env.BASE_SHA; else process.env.BASE_SHA = oldBase;
      if (oldHead === undefined) delete process.env.HEAD_SHA; else process.env.HEAD_SHA = oldHead;
    }
  } finally {
    fs.rmSync(root, {recursive: true, force: true});
  }
});
