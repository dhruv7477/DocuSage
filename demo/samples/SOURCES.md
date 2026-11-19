# Sample documents

Every account on a demo deployment (`DEMO_MODE=true`) gets these files in its "Sample documents" workspace. Both are works of the United States federal government, which are not subject to copyright in the United States (17 U.S.C. 105).

| File | Source | Pages |
|---|---|---|
| `i-9-instructions.pdf` | U.S. Citizenship and Immigration Services, *Instructions for Form I-9, Employment Eligibility Verification* (edition 01/20/25). https://www.uscis.gov/sites/default/files/document/forms/i-9instr.pdf | 8 |
| `nist-csf-2.0.pdf` | National Institute of Standards and Technology, *The NIST Cybersecurity Framework (CSF) 2.0*, NIST CSWP 29, February 26, 2024. https://doi.org/10.6028/NIST.CSWP.29 | 32 |

Both files are unmodified copies, downloaded on 2026-11-17. They are included for demonstration only. Use the current versions from the agencies for anything that matters.

To use different samples, put PDF files in this folder (or point `DEMO_SAMPLES_DIR` elsewhere). The worker adds new files to the sample library when it starts. Accounts created afterwards get them.
