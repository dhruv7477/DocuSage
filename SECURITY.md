# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue or pull request.

- Email **hello@akartalabs.com**, or
- use **Report a vulnerability** on the repository's Security tab (GitHub private vulnerability reporting).

Include what you found, the steps to reproduce it, and the version or commit you tested. You should get a reply within a few working days. Once a fix is released, we are happy to credit you if you wish.

## Supported versions

Only the latest commit on `master` (and the public demo, which runs it) gets security fixes.

## Testing the public demo

If you test the hosted demo, use an account you created yourself and keep to it.

- Do not run automated scanners, load tests or denial-of-service tests against it. Run the stack yourself instead: the README quickstart starts the same setup on your machine.
- Do not try to read, change or delete other users' data.
- Do not upload malware or anything you do not have the right to share.

## How the code is checked

Every pull request and a weekly scheduled run go through:

- dependency checks (Dependabot, dependency review, pip-audit, npm audit, Trivy on the built images),
- static analysis (CodeQL for Python and TypeScript, SonarQube Cloud),
- a secret scan of the full git history (gitleaks),
- an OWASP ZAP baseline scan and API scan against a disposable copy of the demo stack started in CI. Scans never run against a live server.
