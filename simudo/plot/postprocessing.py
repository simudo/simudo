
from scipy.constants import elementary_charge


def df_add_drift_diffusion_terms(df):
    for mobility_key in df.columns:
        before, sep, band = mobility_key.partition('mobility_')
        if not (before == '' and sep): continue
        current_key = 'j_' + band
        if current_key not in df.columns: continue

        df['j_drift_'+band] = j_drift = (
            elementary_charge*df[mobility_key]*df['u_'+band]*df['E'])

        df['j_diffusion_'+band] = df[current_key] - j_drift
