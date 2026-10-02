# Upload this folder to GitHub

Target repository: `https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0`.

1. Sign in to GitHub as the account with push access to the target repository.
2. Open `https://github.com/hezejuyede/Pattern_Analysis_and_Applications_Onion_Reliability_v1.0/upload`.
3. Open this local folder and drag **all of its contents** onto the upload page.
   Drag the contents, not the parent folder and not the ZIP transport copy.
4. Use commit message `Release compact reproducibility repository v1.0`.
5. After the commit completes, confirm that the README and nested directories
   appear at the repository root.
6. Re-run `python scripts/verify_slim_repository.py` after any edit and rebuild
   `SHA256SUMS` if a file changed.

This build is deliberately below GitHub's browser limits of 25 MiB per file and
100 files per upload. The folder itself is the repository root.
