# Community

HakiOS is built by whoever turns up and does the work: people who own a phone and a debugger, people who read
Apple's open source, people who write drivers, tests and documents. This file says how to take part, and keeps the
code and the money apart.

## Taking part

- **You need nothing but a patch.** No account with anybody, no membership, no token. Open an issue, send a pull
  request, or put a result in `docs/` ([CONTRIBUTING.md](CONTRIBUTING.md)).
- **What a contribution is licensed under** is what the file it touches is under: the kernel patch and the files
  that sit in Apple's tree under APSL 2.0, everything else MIT ([LEGAL.md](LEGAL.md), [NOTICE](NOTICE)).
- **A contribution brings nothing of Apple's or a vendor's.** No Darwin file, no firmware, no key, no screenshot of
  either. CI runs `tools/provenance-check.py` and fails a change that does. If you are not sure what counts, ask first.
- **Results are reported as seen.** A test says what ran, on which phone, with which arguments, and what did not.
  "Does not work" and "I did not try" are both useful, and are not the same.

## Ways to help that need no phone

The open configuration (`make qemu-open`) boots in QEMU with nothing of Apple's and nothing of any phone's. Most of
the project is reachable from there: the loader, the configuration tools ([configs.md](docs/configs.md)), the boot
argument catalogue ([boot-args.md](docs/boot-args.md)), the driver reviews ([driver-review/](docs/driver-review)), the
translations, and a port to another SoC starting from its device tree ([devices/](devices)).

## The token

There is a community token, `$HackiOS`. This is what the repository says about it, and it is deliberately short:

- **It is not part of this project's code, licence or governance.** Holding it, buying it or selling it gives no
  right in this repository: no licence, no commit access, no vote on what is merged, no access to a file or a feature.
  Nothing here is gated behind it, and nothing will be.
- **The repository is not a place to promote it.** Issues, pull requests and discussions are for the code. A price,
  a "when", a promise of a return does not belong in them, and will be removed.
- **Nothing in this repository is investment advice or an offer of anything.** What a token is in law depends on what
  it promises and where you are; whoever runs a token is responsible for that, not this repository. If you are thinking
  of buying one, the code is not a reason, and a lawyer who knows your country is a better one.
- **Contributors are judged by their contribution**, not by what they hold.
- **Payments for work** (if the community decides to make any) are for specified work, agreed in advance, by whoever
  agreed to pay. They are not a share of anything and not a reward for holding.

The people who run the token are expected to say, in their own channel and in their own name, who they are, what the
token is meant to do, and what it does not. This file does not speak for them, and they do not speak for the code.

## Conduct

Be exact, be kind about people and hard on claims. No harassment; no doxxing; no pressure to run something
you do not understand. Maintainers may remove what breaks this, and say why.

## Reporting

A security problem, or a file that infringes somebody's rights: open an issue marked as such, or write to the
maintainers listed in the repository, and it will be looked at and, where warranted, removed.
