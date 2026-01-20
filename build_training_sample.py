import os
import time
import numpy as np
import pandas as pd
from astropy.table import Table, join
import sys

from dl import queryClient as qc
from dl import authClient as ac

print("=" * 80)
print("DESI Dwarf Photo-z Training Set Construction (Three Separate Queries)")
print("=" * 80)

# ============================================================================
# CONFIGURATION: Magnitude Limits
# ============================================================================
MAX_R_MAG = 21.2
FLUX_R_MIN = 10.0 ** ((22.5 - MAX_R_MAG) / 2.5)

print(f"\nConfiguration:")
print(f"  Maximum r-band magnitude: {MAX_R_MAG}")
print(f"  Corresponding minimum flux: {FLUX_R_MIN:.6f} nanomaggies")

def apply_quality_cuts(df):
    """Apply quality cuts to DESI galaxy data."""
    df = df.copy()
    print('NUMBER OF OBJECTS BEFORE QUALITY CUTS:', len(df))
    
    # Calculate S/N
    df['snr_g'] = np.where(df['rchisq_g'] < 100, df['flux_g'] * np.sqrt(df['flux_ivar_g']), 0)
    df['snr_r'] = np.where(df['rchisq_r'] < 100, df['flux_r'] * np.sqrt(df['flux_ivar_r']), 0)
    df['snr_z'] = np.where(df['rchisq_z'] < 100, df['flux_z'] * np.sqrt(df['flux_ivar_z']), 0)
    
    # Apply cuts
    mask = (
        # SNR >= 5 in at least 2 bands
        (((df['snr_g'] >= 5) & (df['snr_r'] >= 5)) |
          ((df['snr_g'] >= 5) & (df['snr_z'] >= 5)) |
          ((df['snr_r'] >= 5) & (df['snr_z'] >= 5))) &
        
        # fracflux <= 0.35 in at least 2 bands
        (((df['fracflux_g'] <= 0.35) & (df['fracflux_r'] <= 0.35)) |
          ((df['fracflux_g'] <= 0.35) & (df['fracflux_z'] <= 0.35)) |
          ((df['fracflux_r'] <= 0.35) & (df['fracflux_z'] <= 0.35))) &
        
        # rchisq <= 2.0 in at least 2 bands
        (((df['rchisq_g'] <= 2.0) & (df['rchisq_r'] <= 2.0)) |
          ((df['rchisq_g'] <= 2.0) & (df['rchisq_z'] <= 2.0)) |
          ((df['rchisq_r'] <= 2.0) & (df['rchisq_z'] <= 2.0))) &
        
        # High quality in at least 2 bands
        (((((df['snr_g'] >= 30) | (df['rchisq_g'] <= 0.85)) &
            ((df['snr_r'] >= 30) | (df['rchisq_r'] <= 0.85)))) |
          (((df['snr_g'] >= 30) | (df['rchisq_g'] <= 0.85)) &
           ((df['snr_z'] >= 30) | (df['rchisq_z'] <= 0.85))) |
          (((df['snr_r'] >= 30) | (df['rchisq_r'] <= 0.85)) &
           ((df['snr_z'] >= 30) | (df['rchisq_z'] <= 0.85)))) &
        
        # Color cut
        (2.5 * np.log10(df['flux_r'] / df['flux_g']) > -0.1)
    )
    print('NUMBER OF OBJECTS AFTER QUALITY CUTS:', len(df[mask]))
    return df[mask]

# ============================================================================
# FUNCTION: Identify Survey Target Type
# ============================================================================
def identify_target_survey(spectype, zwarn):
    """Identify survey type from spectype."""
    if pd.isna(spectype):
        return 'UNKNOWN'
    spectype_str = str(spectype).upper()
    if 'GALAXY' in spectype_str:
        return 'GALAXY'
    elif 'STAR' in spectype_str:
        return 'STAR'
    elif 'QSO' in spectype_str:
        return 'QSO'
    else:
        return 'OTHER'

# ============================================================================
# FUNCTION: Create Unique Source Identifier
# ============================================================================
def create_source_id(targetid, survey, program):
    """Create a unique identifier combining targetid, survey, and program."""
    return f"{targetid}_{survey}_{program}"

# ============================================================================
# FUNCTION: Darragh-Ford Color Cuts
# ============================================================================
def flux_to_mag(flux):
    """Convert nanomaggy flux to AB magnitude."""
    return 22.5 - 2.5 * np.log10(flux.clip(lower=1e-9))

def mag_to_flux(mag):
    """Convert AB magnitude to nanomaggy flux."""
    return 10.0 ** ((22.5 - mag) / 2.5)

def flux_to_mag_err(flux, ivar):
    """Compute magnitude error from flux and inverse variance."""
    flux_err = 1.0 / np.sqrt(ivar.clip(lower=1e-9))
    flux_safe = flux.clip(lower=1e-9)
    dmag_dflux = -2.5 / (flux_safe * np.log(10))
    return np.abs(dmag_dflux * flux_err)

def apply_extinction_correction(mag, mw_transmission):
    """Apply Milky Way extinction correction to magnitude."""
    extinction = -2.5 * np.log10(mw_transmission.clip(lower=1e-9))
    return mag - extinction

def compute_surface_brightness_error(mag_r, mag_err_r, shape_r, shape_r_ivar):
    """Compute surface brightness error including errors in both magnitude and radius."""
    shape_r_err = 1.0 / np.sqrt(shape_r_ivar.clip(min=1e-9))
    dmgmu_dshape = 2.5 * (2.0 / (shape_r.clip(min=1e-9) * np.log(10)))
    sigma_mu = np.sqrt(mag_err_r**2 + (dmgmu_dshape * shape_r_err)**2)
    return sigma_mu

def passes_darragh_ford_cuts(df, z_complete_threshold=0.03):
    """Apply the Darragh-Ford et al. color cuts for dwarf galaxy selection."""
    
    if z_complete_threshold == 0.03:
        sb_threshold = 16.8
        color_threshold = 0.99
    elif z_complete_threshold == 0.01:
        sb_threshold = 18.5
        color_threshold = 0.9
    else:
        raise ValueError("z_complete_threshold must be 0.03 or 0.01")
    
    shape_r_arcsec = df['shape_r'].values
    mu_r_eff = df['mag_r_extinction_corrected'].values + 2.5 * np.log10(2 * np.pi * shape_r_arcsec**2)
    
    sigma_mu = compute_surface_brightness_error(
        df['mag_r_extinction_corrected'].values, 
        df['mag_err_r'].values, 
        df['shape_r'].values, 
        df['shape_r_ivar'].values
    )
    
    r_offset = df['mag_r_extinction_corrected'].values - 14
    sb_cut = mu_r_eff + sigma_mu - 0.7 * r_offset > sb_threshold
    
    g_minus_r = df['mag_g_extinction_corrected'].values - df['mag_r_extinction_corrected'].values
    sigma_gr = np.sqrt(df['mag_err_g'].values**2 + df['mag_err_r'].values**2)
    color_cut = g_minus_r - sigma_gr + 0.06 * r_offset < color_threshold
    
    return sb_cut & color_cut

def process_query_results(df):
    """Common processing for all query results with extinction correction."""
    mag_g_raw = flux_to_mag(df['flux_g'])
    mag_r_raw = flux_to_mag(df['flux_r'])
    mag_z_raw = flux_to_mag(df['flux_z'])
    mag_w1_raw = flux_to_mag(df['flux_w1'])
    mag_w2_raw = flux_to_mag(df['flux_w2'])
    
    mag_err_g = flux_to_mag_err(df['flux_g'], df['flux_ivar_g'])
    mag_err_r = flux_to_mag_err(df['flux_r'], df['flux_ivar_r'])
    mag_err_z = flux_to_mag_err(df['flux_z'], df['flux_ivar_z'])
    
    df['mag_g_extinction_corrected'] = apply_extinction_correction(mag_g_raw, df['mw_transmission_g'])
    df['mag_r_extinction_corrected'] = apply_extinction_correction(mag_r_raw, df['mw_transmission_r'])
    df['mag_z_extinction_corrected'] = apply_extinction_correction(mag_z_raw, df['mw_transmission_z'])
    df['mag_w1_extinction_corrected'] = apply_extinction_correction(mag_w1_raw, df['mw_transmission_w1'])
    df['mag_w2_extinction_corrected'] = apply_extinction_correction(mag_w2_raw, df['mw_transmission_w2'])
    
    df['mag_err_g'] = mag_err_g
    df['mag_err_r'] = mag_err_r
    df['mag_err_z'] = mag_err_z
    
    df['mag_g_raw'] = mag_g_raw
    df['mag_r_raw'] = mag_r_raw
    df['mag_z_raw'] = mag_z_raw
    df['mag_w1_raw'] = mag_w1_raw
    df['mag_w2_raw'] = mag_w2_raw
    
    df['g_r'] = df['mag_g_extinction_corrected'] - df['mag_r_extinction_corrected']
    df['r_z'] = df['mag_r_extinction_corrected'] - df['mag_z_extinction_corrected']
    df['z_w1'] = df['mag_z_extinction_corrected'] - df['mag_w1_extinction_corrected']
    
    df['target_survey'] = df['survey']
    
    # Create unique source identifier
    df['source_id'] = df.apply(
        lambda row: create_source_id(row['targetid'], row['survey'], row.get('program', 'unknown')), 
        axis=1
    )
    
    return df

# ============================================================================
# AUTHENTICATE WITH DATA LAB
# ============================================================================
print("\n[Step 0] Authenticating with Data Lab...")

try:
    username = os.environ.get('DATALAB_USERNAME')
    password = os.environ.get('DATALAB_PASSWORD')
    
    if not username or not password:
        print("✗ DATALAB_USERNAME or DATALAB_PASSWORD not set!")
        sys.exit(1)
    
    ac.login(username, password)
    print("✓ Successfully authenticated!")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    raise

# ============================================================================
# STEP 1: Load the Manwadkar+ Dwarf Catalog
# ============================================================================
print("\n[Step 1] Loading and filtering Manwadkar+ Dwarf Catalog...")

manwadkar_path = '/scratch/users/jsomalwa/dwarf_photz/gen_dataset/desi_dwarfs/desi_dr1_dwarf_catalog.fits'

manwadkar = Table.read(manwadkar_path, hdu="MAIN")
fastspec = Table.read(manwadkar_path, hdu="FASTSPEC")

print(f"Loaded {len(manwadkar)} total dwarfs from MAIN HDU")
print(f"Loaded {len(fastspec)} entries from FASTSPEC HDU")

fastspec.rename_columns(['RA', 'DEC'], ['RA_fs', 'DEC_fs'])

print("\nApplying quality masks (DWARF_MASKBIT, ZWARN only)...")
mask_clean = manwadkar['DWARF_MASKBIT'] == 0
mask_zwarn = ~manwadkar['ZWARN']
mask_elg = manwadkar['SAMPLE'] != 'ELG'
mask_clean = mask_clean & mask_zwarn & mask_elg

manwadkar = manwadkar[mask_clean]
manwadkar = join(manwadkar, fastspec, join_type='inner', keys='TARGETID')
print(f"After filtering: {len(manwadkar)} objects")

# Extract Manwadkar targetids with survey and program info
manwadkar_sources = set()
for row in manwadkar:
    targetid = row['TARGETID']
    survey = row['SURVEY']
    program = row['PROGRAM']
    source_id = create_source_id(targetid, survey, program)
    manwadkar_sources.add(source_id)
manwadkar['SOURCE_ID'] = [create_source_id(targetid, survey, program) for targetid, survey, program in manwadkar[['TARGETID','SURVEY','PROGRAM']]]
# Check for duplicates using set
source_ids = manwadkar['SOURCE_ID']
n_duplicates = len(source_ids) - len(set(source_ids))
print(f"Duplicate SOURCE_IDs: {n_duplicates}")

if n_duplicates > 0:
    print("Removing duplicates, keeping first occurrence...")
    seen = set()
    mask = []
    for sid in source_ids:
        if sid not in seen:
            mask.append(True)
            seen.add(sid)
        else:
            mask.append(False)
    manwadkar = manwadkar[mask]
    print(f"After deduplication: {len(manwadkar)} objects (removed {n_duplicates})")


print(f"\nManwadkar sample info:")
print(f"  Redshift range: {manwadkar['Z'].min():.4f} to {manwadkar['Z'].max():.4f}")
print(f"  Stellar mass range: {manwadkar['LOG_MSTAR_SAGA'].min():.2f} to {manwadkar['LOG_MSTAR_SAGA'].max():.2f}")
print(f"  Objects with LOG_MSTAR_SAGA < 9.5: {(manwadkar['LOG_MSTAR_SAGA'] < 9.5).sum()}")

# ============================================================================
# STEP 1B: Upload Manwadkar TARGETIDs to mydb (with survey and program)
# ============================================================================
print("\n[Step 1B] Uploading Manwadkar TARGETIDs to Data Lab mydb...")

# Create a dataframe with targetid, survey, and program for the Manwadkar sample
manwadkar_upload = manwadkar[['TARGETID','SURVEY','PROGRAM']].to_pandas()
manwadkar_upload.columns = ['targetid','survey','program']
manwadkar_upload['survey'] = manwadkar_upload['survey'].str.decode('utf-8')
manwadkar_upload['program'] = manwadkar_upload['program'].str.decode('utf-8')

csv_file = '/tmp/manwadkar_targetids.csv'
manwadkar_upload.to_csv(csv_file, index=False)

table_name = 'manwadkar_targetids'
try:
    qc.query(f"DROP TABLE IF EXISTS mydb://{table_name}")
except:
    pass

print(f"Uploading {len(manwadkar_upload)} entries to mydb.{table_name}...")
qc.mydb_import(table_name, csv_file)
print(f"✓ Successfully uploaded to mydb.{table_name}")


# ============================================================================
# STEP 2: QUERY 1 - Group 1 (Manwadkar)
# ============================================================================
print("\n[Step 2] QUERY 1: Fetching Group 1 (All Manwadkar with good redshifts)...")

sql_g1 = f"""SELECT z.targetid, z.survey, z.program, z.z, z.zerr, z.zwarn, z.spectype, p.ra, p.dec, p.release, p.brickid, p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2, p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z, p.flux_ivar_w1, p.flux_ivar_w2, p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z, p.mw_transmission_w1, p.mw_transmission_w2, p.morphtype, p.shape_r, p.shape_r_ivar, p.sersic, p.fiberflux_r, p.allmask_g, p.allmask_r, p.allmask_z, z.desi_target, z.bgs_target, z.mws_target, z.scnd_target, z.sv1_desi_target, z.sv2_desi_target, z.sv3_desi_target
FROM desi_dr1.zpix AS z 
JOIN desi_dr1.photometry AS p ON z.targetid = p.targetid 
JOIN mydb://{table_name} AS m ON z.targetid = m.targetid AND z.survey = m.survey AND z.program = m.program
WHERE z.zwarn = 0 AND z.z < 1.5 AND p.flux_r > 0 AND p.flux_r > {FLUX_R_MIN} AND p.allmask_r = 0 and z.deltachi2 > 40"""

try:
    print("Downloading Group 1...")
    df_group1 = qc.query(sql_g1, fmt='pandas')
    print(f"✓ Downloaded {len(df_group1)} rows for Group 1")
except Exception as e:
    print(f"✗ Query failed: {e}")
    raise
df_group1 = process_query_results(df_group1)

# ============================================================================
# STEP 3: QUERY 2 - Group 2 (All passing color cuts with redshifts)
# ============================================================================
print("\n[Step 3] QUERY 2: Fetching Group 2 (All galaxies passing low-z color cuts)...")

sql_g2 = f"""SELECT z.targetid, z.survey, z.program, z.z, z.zerr, z.zwarn, z.spectype, p.ra, p.dec, p.release, p.brickid, p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2, p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z, p.flux_ivar_w1, p.flux_ivar_w2, p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z, p.mw_transmission_w1, p.mw_transmission_w2, p.morphtype, p.shape_r, p.shape_r_ivar, p.sersic, p.fiberflux_r, p.allmask_g, p.allmask_r, p.allmask_z, z.desi_target, z.bgs_target, z.mws_target, z.scnd_target, z.sv1_desi_target, z.sv2_desi_target, z.sv3_desi_target, l.rchisq_g, l.rchisq_r, l.rchisq_z, p.fracflux_g, p.fracflux_r, p.fracflux_z 
FROM desi_dr1.zpix AS z 
JOIN desi_dr1.photometry AS p ON z.targetid = p.targetid 
JOIN ls_dr9.tractor as l on p.ls_id = l.ls_id
WHERE z.zwarn = 0 AND z.z < 1.5 AND p.flux_r > 0 AND p.flux_r > {FLUX_R_MIN} AND p.allmask_r = 0 AND z.spectype='GALAXY' and z.deltachi2 > 40 LIMIT 1000000"""

try:
    print("Downloading all objects with redshifts in the low-z color region...")
    df_group2_all = qc.query(sql_g2, fmt='pandas')
    print(f"✓ Downloaded {len(df_group2_all)} rows")
except Exception as e:
    print(f"✗ Query failed: {e}")
    raise

# Filter by Darragh-Ford cuts locally
print("\nFiltering by Darragh-Ford cuts (z < 0.03 completeness)...")
df_group2_all = apply_quality_cuts(df_group2_all)
df_group2_all = process_query_results(df_group2_all)
df_group2_all['passes_df_cuts'] = passes_darragh_ford_cuts(df_group2_all, z_complete_threshold=0.03)
df_group2 = df_group2_all[df_group2_all['passes_df_cuts']].copy()

print(f"After Darragh-Ford cuts: {len(df_group2)} objects")

# Remove Manwadkar objects from Group 2 using source_id
n_removed = len(df_group2[df_group2['source_id'].isin(manwadkar_sources)])
print(f"Removing {n_removed} objects that are already in Group 1...")
df_group2 = df_group2[~df_group2['source_id'].isin(manwadkar_sources)].copy()
print(f"After removing Group 1: {len(df_group2)} objects")

# ============================================================================
# STEP 4: QUERY 3 - Group 3 (Random Background)
# ============================================================================
print("\n[Step 4] QUERY 3: Fetching Group 3 (Random Background Sample)...")

sql_g3 = f"""SELECT z.targetid, z.survey, z.program, z.z, z.zerr, z.zwarn, z.spectype, p.ra, p.dec, p.release, p.brickid, p.flux_g, p.flux_r, p.flux_z, p.flux_w1, p.flux_w2, p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z, p.flux_ivar_w1, p.flux_ivar_w2, p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z, p.mw_transmission_w1, p.mw_transmission_w2, p.morphtype, p.shape_r, p.shape_r_ivar, p.sersic, p.fiberflux_r, p.allmask_g, p.allmask_r, p.allmask_z, z.desi_target, z.bgs_target, z.mws_target, z.scnd_target, z.sv1_desi_target, z.sv2_desi_target, z.sv3_desi_target, l.rchisq_g, l.rchisq_r, l.rchisq_z, p.fracflux_g, p.fracflux_r, p.fracflux_z
FROM desi_dr1.zpix AS z 
JOIN desi_dr1.photometry AS p ON z.targetid = p.targetid 
JOIN ls_dr9.tractor as l on p.ls_id = l.ls_id
WHERE z.zwarn = 0 AND z.z < 1.5 AND p.flux_r > 0 AND p.flux_r > {FLUX_R_MIN} AND p.allmask_r = 0 AND (z.targetid % 20 = 0) AND z.spectype='GALAXY'"""

try:
    print("Downloading random background sample (5% of DR1)...")
    df_group3_all = qc.query(sql_g3, fmt='pandas')
    print(f"✓ Downloaded {len(df_group3_all)} rows")
except Exception as e:
    print(f"✗ Query failed: {e}")
    raise

# Process Group 3
df_group3_all = apply_quality_cuts(df_group3_all)
df_group3_all = process_query_results(df_group3_all)

# Remove objects from Groups 1 and 2 using source_id
n_removed_g1 = len(df_group3_all[df_group3_all['source_id'].isin(manwadkar_sources)])
print(f"Removing {n_removed_g1} objects in Group 1...")
df_group3_all = df_group3_all[~df_group3_all['source_id'].isin(manwadkar_sources)]

n_removed_g2 = len(df_group3_all[df_group3_all['source_id'].isin(df_group2['source_id'])])
print(f"Removing {n_removed_g2} objects in Group 2...")
df_group3 = df_group3_all[~df_group3_all['source_id'].isin(df_group2['source_id'])].copy()

print(f"Final Group 3: {len(df_group3)} objects")

# ============================================================================
# STEP 5: Process All Groups
# ============================================================================
print("\n[Step 5] Processing all groups...")

df_group1 = process_query_results(df_group1)
df_group1['group'] = 'Group_1_Manwadkar'
df_group1['passes_df_cuts'] = passes_darragh_ford_cuts(df_group1, z_complete_threshold=0.03)

df_group2['group'] = 'Group_2_ColorCuts'

df_group3['group'] = 'Group_3_Background'
df_group3['passes_df_cuts'] = passes_darragh_ford_cuts(df_group3, z_complete_threshold=0.03)

# ============================================================================
# STEP 6: Merge All Groups
# ============================================================================
print("\n[Step 6] Merging all groups...")

df_all = pd.concat([df_group1.sample(83333, random_state=42), df_group2.sample(n=83333, random_state=42), df_group3.sample(n=83333, random_state=42)], ignore_index=True)
print(f"Total combined: {len(df_all):,} objects")

# ============================================================================
# STEP 7: Join with Manwadkar metadata
# ============================================================================
print("\n[Step 7] Merging with Manwadkar metadata...")

# Define the specific columns needed from the Manwadkar catalog.
needed_cols = [
    'SOURCE_ID', 'Z', 'LOG_MSTAR_SAGA'
]

# Select only the needed columns from the astropy Table FIRST, then convert to pandas.
manwadkar_metadata = manwadkar[needed_cols].to_pandas()

# Rename columns for clarity in the final merged table.
manwadkar_metadata.columns = [
    'source_id', 'z_manwadkar', 'stellar_mass'
]

# Ensure data types are consistent for merging.
df_merged = df_all.merge(manwadkar_metadata, on='source_id', how='left',validate='m:1')

print(f"Merged {(df_merged['stellar_mass'].notna()).sum():,} objects with Manwadkar metadata")

# ============================================================================
# STEP 8: Add Dwarf Flag Column
# ============================================================================
print("\n[Step 8] Adding dwarf flag column...")

df_merged['is_dwarf_by_mass'] = df_merged['stellar_mass'] < 9.5
n_dwarfs_by_mass = df_merged['is_dwarf_by_mass'].sum()
print(f"Objects flagged as dwarfs (LOG_MSTAR_SAGA < 9.5): {n_dwarfs_by_mass:,}")

# ============================================================================
# STEP 8B: Add Targeting Flags
# ============================================================================
print("\n[Step 8B] Adding targeting flags...")

from desitarget.sv1.sv1_targetmask import desi_mask as sv1_desi_mask
from desitarget.sv3.sv3_targetmask import desi_mask as sv3_desi_mask
from desitarget.targets import desi_mask

# --- BGS (Bright Galaxy Survey) ---
is_bgs_sv1 = (df_merged['sv1_desi_target'] & sv1_desi_mask.BGS_ANY) != 0
is_bgs_sv2 = (df_merged['sv2_desi_target'] & sv3_desi_mask.BGS_ANY) != 0
is_bgs_sv3 = (df_merged['sv3_desi_target'] & sv3_desi_mask.BGS_ANY) != 0
is_bgs_main = (df_merged['bgs_target'] & desi_mask.BGS_ANY) != 0
df_merged['is_targeted_bgs'] = is_bgs_sv1 | is_bgs_sv2 | is_bgs_sv3 | is_bgs_main

# --- LRG (Luminous Red Galaxy) ---
is_lrg_sv1 = (df_merged['sv1_desi_target'] & sv1_desi_mask.LRG) != 0
is_lrg_sv2 = (df_merged['sv2_desi_target'] & sv3_desi_mask.LRG) != 0
is_lrg_sv3 = (df_merged['sv3_desi_target'] & sv3_desi_mask.LRG) != 0
is_lrg_main = (df_merged['desi_target'] & desi_mask.LRG) != 0
df_merged['is_targeted_lrg'] = is_lrg_sv1 | is_lrg_sv2 | is_lrg_sv3 | is_lrg_main

# --- ELG (Emission Line Galaxy) ---
is_elg_sv1 = (df_merged['sv1_desi_target'] & sv1_desi_mask.ELG) != 0
is_elg_sv2 = (df_merged['sv2_desi_target'] & sv3_desi_mask.ELG) != 0
is_elg_sv3 = (df_merged['sv3_desi_target'] & sv3_desi_mask.ELG) != 0
is_elg_main = (df_merged['desi_target'] & desi_mask.ELG) != 0
df_merged['is_targeted_elg'] = is_elg_sv1 | is_elg_sv2 | is_elg_sv3 | is_elg_main

# --- QSO (Quasar) ---
is_qso_sv1 = (df_merged['sv1_desi_target'] & sv1_desi_mask.QSO) != 0
is_qso_sv2 = (df_merged['sv2_desi_target'] & sv3_desi_mask.QSO) != 0
is_qso_sv3 = (df_merged['sv3_desi_target'] & sv3_desi_mask.QSO) != 0
is_qso_main = (df_merged['desi_target'] & desi_mask.QSO) != 0
df_merged['is_targeted_qso'] = is_qso_sv1 | is_qso_sv2 | is_qso_sv3 | is_qso_main

print(f"  Flagged {df_merged['is_targeted_lrg'].sum():,} LRG targets")
print(f"  Flagged {df_merged['is_targeted_elg'].sum():,} ELG targets")
print(f"  Flagged {df_merged['is_targeted_bgs'].sum():,} BGS targets")
print(f"  Flagged {df_merged['is_targeted_qso'].sum():,} QSO targets")

# ============================================================================
# STEP 10: Select Final Columns
# ============================================================================
print("\n[Step 10] Selecting final columns...")

final_columns = [
    'targetid', 'ra', 'dec', 'brickid', 'release',
    'z', 'zerr', 'zwarn', 'spectype', 'survey', 'program', 'group', 'target_survey', 'source_id',
    'flux_g', 'flux_r', 'flux_z', 'flux_w1', 'flux_w2',
    'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z', 'flux_ivar_w1', 'flux_ivar_w2',
    'mag_g_raw', 'mag_r_raw', 'mag_z_raw', 'mag_w1_raw', 'mag_w2_raw',
    'mag_g_extinction_corrected', 'mag_r_extinction_corrected', 'mag_z_extinction_corrected', 
    'mag_w1_extinction_corrected', 'mag_w2_extinction_corrected',
    'mag_err_g', 'mag_err_r', 'mag_err_z',
    'g_r', 'r_z', 'z_w1',
    'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z', 
    'mw_transmission_w1', 'mw_transmission_w2',
    'morphtype', 'shape_r', 'shape_r_ivar', 'sersic', 'fiberflux_r',
    'passes_df_cuts',
    'z_manwadkar', 'stellar_mass', 'is_dwarf_by_mass', 
    'is_targeted_lrg', 'is_targeted_elg', 'is_targeted_qso', 'is_targeted_bgs'
]
df_final = df_merged[final_columns].copy()

# ============================================================================
# STEP 11: Save Output
# ============================================================================
print("\n[Step 11] Saving output...")

output_csv = 'desi_dwarf_training_sample_full.csv.gz'
output_fits = 'desi_dwarf_training_sample_full.fits'

df_final.to_csv(output_csv, index=False, compression='gzip')
print(f"✓ Saved CSV: {output_csv}")

from astropy.table import Table as AstropyTable
fits_table = AstropyTable.from_pandas(df_final)
fits_table.write(output_fits, overwrite=True)
print(f"✓ Saved FITS: {output_fits}")

# ============================================================================
# STEP 12: Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nTotal objects: {len(df_final):,}")

print(f"\nUnique source_ids (targetid + survey + program combinations): {df_final['source_id'].nunique():,}")
print(f"Duplicate source_ids: {len(df_final) - df_final['source_id'].nunique():,}")

print(f"\nBreakdown by group:")
for group_name in ['Group_1_Manwadkar', 'Group_2_ColorCuts', 'Group_3_Background']:
    count = (df_final['group'] == group_name).sum()
    pct = 100.0 * count / len(df_final)
    print(f"  {group_name}: {count:,} ({pct:.1f}%)")

print(f"\nTarget Survey Types:")
for survey in sorted(df_final['survey'].unique()):
    count = (df_final['survey'] == survey).sum()
    pct = 100.0 * count / len(df_final)
    print(f"  {survey}: {count:,} ({pct:.1f}%)")

print(f"\nTarget Program Types:")
for program in sorted(df_final['program'].unique()):
    count = (df_final['program'] == program).sum()
    pct = 100.0 * count / len(df_final)
    print(f"  {program}: {count:,} ({pct:.1f}%)")

print(f"\nDwarf Mass Flag (is_dwarf_by_mass):")
print(f"  Objects with LOG_MSTAR_SAGA < 9.5: {df_final['is_dwarf_by_mass'].sum():,}")
print(f"  Objects with LOG_MSTAR_SAGA >= 9.5 or no mass info: {(~df_final['is_dwarf_by_mass']).sum():,}")

print(f"\nRedshift distribution:")
print(f"  Min Z: {df_final['z'].min():.6f}")
print(f"  Max Z: {df_final['z'].max():.6f}")
print(f"  Median Z: {df_final['z'].median():.6f}")
print(f"  Mean Z: {df_final['z'].mean():.6f}")

print(f"\nMagnitude distribution (r-band, no ext corr):")
print(f"  Min mag_r: {df_final['mag_r_raw'].min():.2f}")
print(f"  Max mag_r: {df_final['mag_r_raw'].max():.2f}")
print(f"  Median mag_r: {df_final['mag_r_raw'].median():.2f}")

print(f"\nMagnitude distribution (r-band):")
print(f"  Min mag_r: {df_final['mag_r_extinction_corrected'].min():.2f}")
print(f"  Max mag_r: {df_final['mag_r_extinction_corrected'].max():.2f}")
print(f"  Median mag_r: {df_final['mag_r_extinction_corrected'].median():.2f}")

print(f"\nColor distribution (g-r):")
print(f"  Min g-r: {df_final['g_r'].min():.2f}")
print(f"  Max g-r: {df_final['g_r'].max():.2f}")
print(f"  Median g-r: {df_final['g_r'].median():.2f}")

print(f"\nWith Manwadkar metadata: {(df_final['stellar_mass'].notna()).sum():,} objects")

print(f"\nDarragh-Ford Cut Statistics:")
print(f"  Objects passing cuts: {(df_final['passes_df_cuts']).sum():,}")
print(f"  Objects NOT passing cuts: {(~df_final['passes_df_cuts']).sum():,}")

print(f"\nTargeting Flag Breakdown (on final sample):")
print(f"  Targeted as LRG: {df_final['is_targeted_lrg'].sum():,}")
print(f"  Targeted as ELG: {df_final['is_targeted_elg'].sum():,}")
print(f"  Targeted as BGS: {df_final['is_targeted_bgs'].sum():,}")
print(f"  Targeted as QSO (before spectype cut): {df_final['is_targeted_qso'].sum():,}")


print("\n" + "=" * 80)
print("SUCCESS! Training sample ready for machine learning.")
print("=" * 80)

print("\nSample of 5 random rows (Group 1, dwarfs by mass):")
group1_dwarfs = df_final[(df_final['group'] == 'Group_1_Manwadkar') & (df_final['is_dwarf_by_mass'])]
if len(group1_dwarfs) > 0:
    sample = group1_dwarfs.sample(min(5, len(group1_dwarfs)), random_state=42)[
        ['targetid', 'survey', 'program', 'source_id', 'z', 'mag_r_extinction_corrected', 'g_r', 'group', 'target_survey', 'stellar_mass']
    ]
    print(sample)

print("\n" + "=" * 80)
