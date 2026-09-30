# UT CS 378 — GPU node request script

`gpulease.py` gets your group two AWS **g4dn.xlarge** nodes (one NVIDIA T4
each) on demand. Your group shares **80 GPU-hours** for the assignment, counted
per node — see [The budget](#the-budget).

Your access token arrives by email. **It is yours personally — do not share
it.** Anyone holding it can start and stop your group's nodes and destroy what
is on them. No AWS account is needed.

---

## Setup

`gpulease.py` is in `resource_request/` in your repository and needs only
**Python 3.9+** (`python3 --version` to check). Run the commands below from your
repository root.

> **Windows:** tick *"Add python.exe to PATH"* when installing Python, and type
> **`py`** wherever this file says `python3`.

Save your token, once per machine:

```bash
python3 resource_request/gpulease.py login <your-token>
```

If you lose it, ask the course staff to reissue it.

## Start, check, stop

```
python3 resource_request/gpulease.py start     bring your group's nodes up
python3 resource_request/gpulease.py status    lease, budget, and an ssh command per node
python3 resource_request/gpulease.py stop      destroy the nodes and stop the charges
```

- **`start`** takes a few minutes and returns once the nodes accept ssh. If it
  times out, run `status` a minute later. If your group already has a session,
  `start` returns that one.
- **`status`** lists `node0` and `node1` with their ssh commands, when the lease
  ends, and your remaining budget.
- **`stop`** destroys both nodes. Read the next section first.

## Your nodes are temporary

`stop`, the end of your lease, and the assignment deadline all **destroy the
nodes and their disks** — no snapshot, no undo. Each session starts from a fresh
copy of the course image, so you clone your repo onto both nodes and set up again.

**Push to git before you stop.** For anything not in git, copy it off each node
first and check it arrived:

```bash
# on the node
tar czf /tmp/hw.tar.gz --exclude=__pycache__ --exclude='*.pt' -C ~ assignment2

# on your own machine, using the key and host that `status` printed
scp -i ~/.ssh/gpulease_<group> ubuntu@<host>:/tmp/hw.tar.gz .
```

## The budget

Your group shares **80 GPU-hours**, charged by wall-clock time while the nodes
are up, used or not. With both nodes up you spend two GPU-hours per hour, so
the budget lasts **40 hours** of running time. Two nodes left on overnight cost
about sixteen.

- **Start as often as you like.** Only running time is charged.
- **`stop` as soon as you are done.** It is the only thing that stops the meter;
  there is no idle timeout.
- If `status` says `(all the gpu-hours you have left)`, your budget is ending
  the lease, not a fault. Below about 15 minutes, `start` refuses.
- When the budget is gone, you are done unless the course staff extend it.

**Everything ends at the assignment deadline**, whatever budget is left.

## One lease per group

Whoever runs `start` first brings the nodes up; everyone else's `start` returns
the same nodes and key. **`stop` ends the session for your whole group** — tell
them first.

The two nodes share one private network (no other group can reach it) and
accept the same key. From either node, `ssh node0` or `ssh node1` hops to the
other. On each node, `MASTER_ADDR`, `MASTER_PORT`, `NNODES`, and `NODE_RANK` are
exported in every login shell, and the peers are listed in
`/etc/gpulease/peers` and `/etc/hosts`. These appear a few minutes after the
nodes boot. Until then, logging in prints `cluster info ... is not available
yet`, so wait a minute and log in again.

---

## Connecting from VS Code

Remote-SSH reads `~/.ssh/config`, not the commands `status` prints. Add an
entry per node:

```
Host cs378-node0
    HostName <node0's address from status>
    User ubuntu
    IdentityFile ~/.ssh/gpulease_<group>
    IdentitiesOnly yes
    ForwardAgent yes
    ServerAliveInterval 30
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host cs378-node1
    HostName <node1's address from status>
    # ...same remaining lines as cs378-node0
```

Then *Remote-SSH: Connect to Host…* → `cs378-node0` (and `cs378-node1` in a
second window).

- **Addresses change every session.** After each `start`, paste the new ones
  from `status`. A stale address is the usual reason Remote-SSH hangs.
- **Keep the last two lines.** Host keys change every session; without them
  you will hit `REMOTE HOST IDENTIFICATION HAS CHANGED`. On Windows use
  `UserKnownHostsFile NUL`.
- **`ForwardAgent yes`** lets you `ssh node1` from node0. It needs your key
  loaded locally: `ssh-add ~/.ssh/gpulease_<group>`. On Linux or WSL, if that
  says `Could not open a connection to your authentication agent`, run
  `eval "$(ssh-agent -s)"` first. That lasts for the current terminal only.
- The first connect each session is slow while VS Code installs its server.

> **Windows + WSL:** if you run `gpulease.py` in WSL but VS Code on Windows,
> they don't share `.ssh`. Either run the script in PowerShell (`py
> resource_request/gpulease.py start`), or after each `start` run
> `cp ~/.ssh/gpulease_<group> /mnt/c/Users/<you>/.ssh/` and point
> `IdentityFile` at `C:/Users/<you>/.ssh/gpulease_<group>`.

## If something goes wrong

| What you see | What to do |
|---|---|
| `Your CLI is out of date` | Download `gpulease.py` from the course page again. |
| `Bad or missing token` | Run `login` again with the whole token. |
| `cannot reach the lease service` | Check your network; if it persists, ask on the forum. |
| `Launch failed (…)` | Tell the course staff, quoting the `ref` in the message. |
| ssh: `Permission denied (publickey)` | Run `start` again — it rewrites the key file. |
| ssh: `error in libcrypto` | Download the script again, then `start`. |
| Remote-SSH hangs | Stale `HostName`. Update it from `status`. |
| `ssh node1` from node0 is denied | Locally, run `eval "$(ssh-agent -s)"` (Linux/WSL) and `ssh-add ~/.ssh/gpulease_<group>`, then reconnect with `ssh -A`. |
| Login says `cluster info ... is not available yet` | Normal for the first few minutes after `start`. Log in again shortly. |
| `NODE_RANK` etc. empty in an open terminal | Run `source /etc/profile.d/gpulease-cluster.sh`. |
| Still no `/etc/gpulease/peers` 15 min after `start` | Tell the course staff, and include the output of `journalctl -u gpulease-peers`. |

A slow `start` is normal. If `start` refuses, the message says whether it's the
budget or the deadline.

## Files on your machine

```
~/.config/gpulease/credentials    your saved token
~/.ssh/gpulease_<group>           this session's key, replaced every session
```

Both are readable only by you. `python3 resource_request/gpulease.py --help`
prints a summary.
