import json
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_script(workflow, step, payload, environment=None, cwd=None):
    lines = (ROOT / '.github' / 'workflows' / workflow).read_text().splitlines()
    start = lines.index(f'      - name: {step}')
    script = lines.index('          script: |', start) + 1
    body = []
    for line in lines[script:]:
        if line and not line.startswith('            '):
            break
        body.append(line[12:])
    wrapper = '''
      const input = JSON.parse(process.argv[1]);
      const outputs = {};
      const context = input.context;
      const comments = input.comments || [];
      const github = {
        rest: { pulls: { get: async () => ({ data: input.pr }) },
                issues: { listComments: {},
                  createComment: async ({body}) => comments.push({user: {login: 'github-actions[bot]'}, body}),
                  updateComment: async ({comment_id, body}) => { comments.find(c => c.id === comment_id).body = body; }
                } },
        paginate: async () => comments
      };
      const core = { setOutput: (key, value) => { outputs[key] = String(value); },
                     info: () => {}, warning: () => {} };
      Object.assign(process.env, input.environment || {});
      (async () => {\n''' + '\n'.join(body) + '''\n})()
        .then(() => console.log(JSON.stringify({...outputs, _comments: comments})))
        .catch(error => { console.error(error); process.exit(1); });
    '''
    result = subprocess.run(['node', '-e', wrapper, json.dumps({**payload, 'environment': environment or {}})],
                            check=True, text=True, capture_output=True, cwd=cwd)
    return json.loads(result.stdout)


class SpecReviewTests(unittest.TestCase):
    fingerprint = 'a' * 64
    personas = ('spec-ambiguity', 'spec-operability', 'spec-security')

    def comment(self, persona, fingerprint=None, author='github-actions[bot]'):
        return {'user': {'login': author}, 'body': f'<!-- agentic-review:{persona} -->\n<!-- spec-fingerprint:{fingerprint or self.fingerprint} -->\nreport'}

    def decision(self, comments, fingerprint=None, changed='true'):
        return run_script('spec-review.yml', 'Check previous reviews', {
            'context': {'repo': {'owner': 'org', 'repo': 'project'}, 'issue': {'number': 42}},
            'comments': comments,
        }, {'SPEC_FINGERPRINT': fingerprint or self.fingerprint, 'SPEC_CHANGED': changed})['review']

    def test_completed_unchanged_review_is_skipped(self):
        self.assertEqual(self.decision([self.comment(p) for p in self.personas]), 'false')

    def test_changed_spec_or_incomplete_review_runs(self):
        comments = [self.comment(p) for p in self.personas]
        self.assertEqual(self.decision(comments, fingerprint='b' * 64), 'true')
        self.assertEqual(self.decision(comments[:-1]), 'true')
        self.assertEqual(self.decision([self.comment(p, author='alice') for p in self.personas]), 'true')

    def test_empty_spec_diff_is_skipped(self):
        self.assertEqual(self.decision([], changed='false'), 'false')

    def test_published_reports_record_a_reusable_fingerprint(self):
        context = {'repo': {'owner': 'org', 'repo': 'project'}, 'issue': {'number': 42},
                   'payload': {'pull_request': {'head': {'sha': 'head'}}}}
        with tempfile.TemporaryDirectory() as directory:
            for persona in self.personas:
                report = pathlib.Path(directory) / 'reports' / persona / f'{persona}.txt'
                report.parent.mkdir(parents=True)
                report.write_text('review findings')
            published = run_script('spec-review.yml', 'Publish reviews',
                                   {'context': context, 'pr': {'head': {'sha': 'head'}}},
                                   {'SPEC_FINGERPRINT': self.fingerprint}, cwd=directory)
            self.assertEqual(self.decision(published['_comments']), 'false')

    def test_fingerprint_ignores_code_changes_but_tracks_spec_changes(self):
        lines = (ROOT / '.github/workflows/spec-review.yml').read_text().splitlines()
        start = lines.index('      - name: Fingerprint specification diff')
        start = lines.index('        run: |', start) + 1
        commands = []
        for line in lines[start:]:
            if line and not line.startswith('          '):
                break
            commands.append(line[10:])
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory)
            def git(*args):
                return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
            git('init', '-q')
            git('config', 'user.email', 'test@example.com')
            git('config', 'user.name', 'Test')
            (repo / 'specs').mkdir()
            (repo / 'specs/example.md').write_text('original\n')
            git('add', '.')
            git('commit', '-qm', 'base')
            base = git('rev-parse', 'HEAD')
            (repo / 'specs/example.md').write_text('reviewed\n')
            git('commit', '-qam', 'spec change')
            def fingerprint():
                output = repo / 'output'
                output.write_text('')
                env = {**os.environ, 'BASE_SHA': base, 'HEAD_SHA': git('rev-parse', 'HEAD'),
                       'RUNNER_TEMP': directory, 'GITHUB_OUTPUT': str(output)}
                subprocess.run(['bash', '-eo', 'pipefail', '-c', '\n'.join(commands)], cwd=repo, env=env, check=True)
                return dict(line.split('=', 1) for line in output.read_text().splitlines())
            reviewed = fingerprint()
            (repo / 'code.txt').write_text('code\n')
            git('add', 'code.txt')
            git('commit', '-qm', 'code change')
            self.assertEqual(fingerprint(), reviewed)
            self.assertEqual(reviewed['changed'], 'true')
            (repo / 'specs/example.md').write_text('changed again\n')
            git('commit', '-qam', 'another spec change')
            self.assertNotEqual(fingerprint()['fingerprint'], reviewed['fingerprint'])


class PullRequestReviewTests(unittest.TestCase):
    def request(self, body='/review', association='MEMBER', issue_is_pr=True, draft=False,
                head_repo='org/project', state='open'):
        issue = {'number': 42}
        if issue_is_pr:
            issue['pull_request'] = {'url': 'https://api.github.com/repos/org/project/pulls/42'}
        context = {'repo': {'owner': 'org', 'repo': 'project'}, 'issue': {'number': 42},
                   'payload': {'issue': issue, 'comment': {'body': body, 'author_association': association}}}
        pr = {'state': state, 'draft': draft, 'body': 'spec declaration',
              'base': {'sha': 'base'}, 'head': {'sha': 'head', 'repo': {'full_name': head_repo}}}
        return run_script('pr-review.yml', 'Resolve review request', {'context': context, 'pr': pr})

    def test_review_command_resolves_current_pr(self):
        output = self.request(body=' /review\n')
        self.assertEqual(output['eligible'], 'true')
        self.assertEqual((output['base'], output['head'], output['body']), ('base', 'head', 'spec declaration'))

    def test_other_comments_and_untrusted_requests_are_ineligible(self):
        for options in ({'body': 'please /review'}, {'body': '/review extra'},
                        {'association': 'NONE'}, {'issue_is_pr': False}, {'draft': True},
                        {'head_repo': 'other/fork'}, {'state': 'closed'}):
            with self.subTest(options=options):
                self.assertNotEqual(self.request(**options).get('eligible'), 'true')


if __name__ == '__main__':
    unittest.main()
