# From FASTQ to Spliced + Unspliced + Spatial Matrix
### Sample: E13 Mouse Head · Platform: Open-ST · Tool: Spacemake 0.9.5

---

## Overview

```
Raw FASTQs (R1 + R2)
      │
      ▼
[1] Barcode Tagging + Quality Trimming      ~9.5 min   → unaligned CRAM
      │
      ▼
[2] Spatial Barcode Correction             ~28 min    → corrected CRAM
      │
      ▼
[3] PhiX Contamination Removal (bowtie2)   ~21 min    → not_phiX CRAM
      │
      ▼
[4] rRNA Removal (bowtie2)                 ~variable  → not_rRNA CRAM
      │
      ▼
[5] Genome Alignment (STAR)                ~hours     → final BAM
      │
      ▼
[6] Per-Tile DGE Matrix Creation           ~parallel  → sparse count matrices
      │
      ▼
[7] h5ad Assembly                          ~minutes   → AnnData object
                                                         with spliced, unspliced,
                                                         and spatial coordinates
```

---

## Input Files

| File | Description |
|---|---|
| `e13_mouse_head_R1_001.fastq.gz` (3.5GB) | Read 1 — contains spatial barcode + UMI |
| `e13_mouse_head_R2_001.fastq.gz` (12GB) | Read 2 — contains cDNA sequence |
| `e13_mouse_head_reseq_R1_001.fastq.gz` (7.6GB) | Resequencing R1 (merged with above) |
| `e13_mouse_head_reseq_R2_001.fastq.gz` (17GB) | Resequencing R2 (merged with above) |
| `GRCm39vM30.genome.fa` (2.6GB) | Mouse genome reference (GRCm39) |
| `gencodevM30.annotation.gtf` (868MB) | Gene annotation (GENCODE M30) |
| `mouse.rRNA.fa` | Ribosomal RNA sequences for filtering |
| `phiX.fa` | PhiX174 spike-in for filtering |
| `fc_1_L3_tile_*.txt.gz` (12 tiles) | Spatial barcode whitelists per flow cell tile |

### Open-ST Read Layout

```
R1: [skip 2bp][─── 25bp spatial barcode ───][── 9bp UMI ──][ ... ]
R2: [─────────────── cDNA sequence ──────────────────────────────]
```

The spatial barcode in R1 encodes which tile and x/y coordinate on the
flow cell the RNA molecule was captured from. The 12 tiles here correspond
to a single lane region of the patterned flow cell.

---

## Step 1 — Barcode Tagging + Trimming

**Tool:** `spacemake.bin.fastq_to_uBAM`

Both R1+R2 pairs (original + resequencing) are processed together as a merged
input. For each read pair:

1. The spatial barcode is extracted from `r1[2:27]` and the UMI from `r2[0:9]`
   and written as BAM tags into an unaligned CRAM file
2. Quality trimming removes 3′ low-quality bases from R2 (NextSeq Q-score ≥ 25)
3. The SMART adapter (`AAGCAGTGGTATCAACGCAGAGTGAATGGG`) is trimmed from R2
4. PolyA tails are trimmed from R2

**Stats from this run:**
- Input: **569.6M reads**
- Output: **550.6M reads kept (96.66%)**
- Trimmed for polyA: 102.2M reads
- Trimmed for SMART adapter: 101.6k reads

---

## Step 2 — Spatial Barcode Correction

**Tool:** `cb_index_corrected_sample`

The raw 25bp barcode extracted from R1 rarely matches the whitelist perfectly due
to sequencing errors. Each barcode is fuzzy-matched (up to 1 mismatch allowed)
against the 12 tile whitelists. Reads that match are tagged with the corrected
barcode and the tile ID.

This step assigns each molecule to a specific (x, y) coordinate on the flow cell,
which becomes the spatial coordinate in the final matrix.

---

## Step 3 — PhiX Removal

**Tool:** bowtie2

Reads in the corrected CRAM are aligned to the PhiX174 genome. PhiX is a
bacteriophage added to Illumina sequencing runs as an internal calibration
control — it carries no biological signal and must be removed.

- Aligned reads → `phiX.bowtie2.cram` (discarded)
- Unaligned reads → `not_phiX.bowtie2.cram` (kept, ~21 min)

---

## Step 4 — rRNA Removal

**Tool:** bowtie2

Ribosomal RNA is the dominant RNA species in a cell by mass (~80%) but carries
no information about gene expression. Remaining reads are aligned to `mouse.rRNA.fa`.

- Aligned reads → `rRNA.bowtie2.cram` (discarded)
- Unaligned reads → `not_rRNA.bowtie2.cram` (kept)

**Map strategy defined in `project_df.csv`:**
```
bowtie2:phiX -> bowtie2:rRNA -> STAR:genome:final
```

---

## Step 5 — Genome Alignment (STAR)

**Tool:** STAR ≥ 2.7.1a

Clean reads are aligned to the full GRCm39 mouse genome using the pre-built
STAR index (`species_data/mouse/genome/star_index/`). STAR uses the splice junction
database from `gencodevM30.annotation.gtf` to align reads across exon-exon
junctions.

**This step is the source of the spliced/unspliced distinction:**

- A read that aligns **entirely within an exon** → counted as **spliced (mature mRNA)**
- A read that aligns **within an intron** → counted as **unspliced (pre-mRNA)**

This works because `count_intronic_reads: true` is set in the `openst` run mode
in `config.yaml`. Spacemake instructs STAR's `--soloFeatures` to output both
`Gene` (exonic) and `GeneFull` (exon+intron) count matrices. The unspliced
counts are derived as: `GeneFull − Gene`.

---

## Step 6 — Per-Tile DGE Matrix Creation

**Tools:** `filter_mm_reads`, `create_dge`, `make_whitelist_for_dge`

The final BAM is split by the 12 flow cell tiles. For each tile, a sparse
Digital Gene Expression (DGE) matrix is built independently and then merged.

Three UMI cutoff thresholds are applied in parallel to generate matrices at
different levels of stringency:

| UMI cutoff | Interpretation |
|---|---|
| 100 UMI | Permissive — retains more cells, more noise |
| 250 UMI | Balanced (default for Open-ST) |
| 500 UMI | Stringent — higher quality cells only |

Multi-mapping reads (`count_mm_reads: true`) are included and resolved
proportionally across genes.

---

## Step 7 — h5ad Assembly + Spatial Meshing

**Tool:** `create_h5ad_dge`

The per-tile sparse matrices are combined into a single AnnData (`.h5ad`) object.
Spatial coordinates from the barcode whitelist are joined onto each cell barcode.

Because Open-ST tiles represent sub-millimeter patches, the data is then
**spatially meshed** into a hexagonal grid:

```yaml
# from config.yaml openst run_mode:
mesh_type: hexagon
mesh_spot_diameter_um: 7      # 7 micron hexagonal bins
mesh_spot_distance_um: 7      # touching (no gap between bins)
```

Each hexagonal bin aggregates all barcodes within its area. This regularises
the irregular bead capture positions into a uniform grid comparable to Visium
(but at ~8x higher resolution, since Visium spots are 55μm).

### Final AnnData Structure

```
AnnData object
  obs:         cells/spots (spatial bins)
    .obsm['spatial']   →  (x, y) coordinates in microns
  var:         genes (GRCm39 / GENCODE M30)
  layers:
    'spliced'          →  exonic UMI counts   (mature mRNA)
    'unspliced'        →  intronic UMI counts (pre-mRNA)
  X:                   →  total counts (spliced + unspliced)
```

This is the direct input format for RNA velocity tools (scVelo, veloVI) and
for the spatial mechanistic model in this project.

---

## Reference Versions

| Reference | Version |
|---|---|
| Mouse genome | GRCm39 (mm39) |
| Gene annotation | GENCODE M30 |
| Spacemake | 0.9.5 |
| STAR | ≥ 2.7.1a |
| bowtie2 | ≥ 2.3.4 |
| Drop-seq tools | 2.5.1 |

---

## Reproducing This Run

```bash
# 1. Initialise spacemake project (done once)
spacemake init

# 2. Add the sample (done once)
spacemake projects add_sample \
  --project-id openst_demo \
  --sample-id openst_demo_e13_mouse_head \
  --R1 ../e13_mouse_head_R1_001.fastq.gz ../e13_mouse_head_reseq_R1_001.fastq.gz \
  --R2 ../e13_mouse_head_R2_001.fastq.gz ../e13_mouse_head_reseq_R2_001.fastq.gz \
  --species mouse \
  --run-mode openst \
  --barcode-flavor openst \
  --puck openst \
  --puck-barcode-file fc_1_L3_tile_2416.txt.gz [... all 12 tiles]

# 3. Run the pipeline
spacemake run --cores 32 --keep-going
```
