import urllib.request

urls = [
    "https://raw.githubusercontent.com/KempnerInstitute/overcomplete/main/overcomplete/sae/rasae.py",
    "https://raw.githubusercontent.com/KempnerInstitute/overcomplete/main/overcomplete/sae/mp_sae.py",
    "https://raw.githubusercontent.com/KempnerInstitute/overcomplete/main/overcomplete/sae/omp_sae.py"
]

out_file = "interpretability_analysis/custom_saes.py"

with open(out_file, "w", encoding="utf-8") as f:
    f.write("# Copied from the overcomplete library (https://github.com/KempnerInstitute/overcomplete)\n\n")
    for url in urls:
        print(f"Downloading {url}...")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            code = response.read().decode('utf-8')
            f.write(code)
            f.write("\n\n")

print(f"Successfully wrote to {out_file}")
