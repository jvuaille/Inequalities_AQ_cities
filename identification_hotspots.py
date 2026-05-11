# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import List, Dict

# Code created with the help of Copilot
# Flag clusters only within the above-median group per variable, using the
# 'very next lower' comparison:
#   Sort above-median values (strict '> median') in descending order.
#   For positions 0..n-2: compare to the next lower within the above group.
#   For position n-1 (the minimum above-median value): compare to the next
#   lower overall (max among values strictly less than its value), which may
#   sit in the below-median group.
#   If there is only one above-median cluster, do the same 'next lower overall'
#   comparison for it.
#   Flag True if (v_curr - v_next_lower) / max(|v_next_lower|, eps) >= pct_threshold.
#    Clusters at/under the median are never flagged for that variable.

def flag_above_median_next_gap_v2(
    df: pd.DataFrame,
    cluster_col: str,
    pollutant_cols: List[str],
    socio_cols: List[str],
    value_cols: List[str] = None,
    pct_threshold: float = 0.05,
    eps: float = 1e-12
) -> Dict[str, object]:

    # Select variables
    if value_cols is None:
        value_cols = list(dict.fromkeys(pollutant_cols + socio_cols))
    missing = [c for c in value_cols + [cluster_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # Variables × clusters (transpose)
    work = df[[cluster_col] + value_cols].copy()
    work[cluster_col] = work[cluster_col].astype(str)
    data_t = work.set_index(cluster_col)[value_cols].T  # rows: variables, cols: clusters

    # Medians per variable
    medians = data_t.median(axis=1)

    # Above-median mask (strict)
    above_flags_t = data_t.gt(medians, axis=0)  # variables × clusters (True if > median)

    # Outputs
    next_gap_flags_t  = pd.DataFrame(False, index=data_t.index, columns=data_t.columns)
    rel_gap_to_next_t = pd.DataFrame(np.nan,   index=data_t.index, columns=data_t.columns)

    for var in data_t.index:
        s_all = data_t.loc[var].astype(float)     # values for this variable (Series indexed by clusters)
        m     = medians.loc[var]

        # Above-median values sorted descending
        s_above = s_all[s_all > m].sort_values(ascending=False)

        if len(s_above) == 0:
            # No above-median clusters: nothing to do
            continue

        # Helper to compute the 'next lower overall' for a given value
        def next_lower_overall(v_curr: float) -> float:
            """Return the maximum value strictly less than v_curr, or np.nan if none."""
            below_strict = s_all[s_all < v_curr]
            return below_strict.max() if not below_strict.empty else np.nan

        vals = s_above.values
        idxs = s_above.index.to_list()

        # Iterate through above-median list
        for i, cid in enumerate(idxs):
            v_curr = vals[i]

            if i < len(vals) - 1:
                # Compare to the very next lower within the above group
                v_next = vals[i + 1]
            else:
                # Minimum above-median element: compare to next lower OVERALL
                v_next = next_lower_overall(v_curr)

            # If there's no strictly lower value at all, cannot flag
            if pd.isna(v_next):
                rel_gap_to_next_t.at[var, cid] = np.nan
                continue

            rel_gap = (v_curr - v_next) / max(abs(v_next), eps)
            rel_gap_to_next_t.at[var, cid] = rel_gap
            if rel_gap >= pct_threshold:
                next_gap_flags_t.at[var, cid] = True

        
       # ---------- Employment-specific LOW-side rule ----------
        # For MEAN_EMP_share, also flag <= median clusters if they are >= pct_threshold lower than:
        #   (a) the very next higher value in the BELOW-MEDIAN group (<= median), OR
        #   (b) the first value strictly above median (if any).
        if var == "MEAN_EMP_share":
            s_below = s_all[s_all <= m].sort_values(ascending=True)
            if len(s_below) > 0:
                # (b) first value above the median
                above_group = s_all[s_all > m]
                first_above = above_group.min() if not above_group.empty else np.nan

                for cid, v_curr in s_below.items():
                    # (a) very next higher value RESTRICTED to the below-median group
                    #     i.e., smallest value in s_below that is strictly greater than current
                    higher_within_below = s_below[s_below > v_curr]
                    v_next_higher_below = higher_within_below.min() if not higher_within_below.empty else np.nan

                    gap1 = np.nan
                    if pd.notna(v_next_higher_below):
                        # relative gap measured from the lower baseline (current)
                        gap1 = (v_next_higher_below - v_curr) / max(abs(v_curr), eps)

                    gap2 = np.nan
                    if pd.notna(first_above):
                        gap2 = (first_above - v_curr) / max(abs(v_curr), eps)

                    best_gap = np.nanmax([gap1, gap2])
                    if pd.notna(best_gap):
                        existing = rel_gap_to_next_t.at[var, cid]
                        if pd.isna(existing) or best_gap > existing:
                            rel_gap_to_next_t.at[var, cid] = best_gap
                        if best_gap >= pct_threshold:
                            next_gap_flags_t.at[var, cid] = True

    # Per-cluster summaries
    poll_flags = next_gap_flags_t.loc[next_gap_flags_t.index.isin(pollutant_cols)]
    socio_flags = next_gap_flags_t.loc[next_gap_flags_t.index.isin(socio_cols)]

    any_5pct_pollution = poll_flags.any(axis=0).rename("any_5pct_pollution")
    any_5pct_socio     = socio_flags.any(axis=0).rename("any_5pct_socio")
    both_types         = (any_5pct_pollution & any_5pct_socio).rename("any_5pct_both_types")

    return {
        "data_t": data_t,
        "medians": medians,
        "above_median_flags_t": above_flags_t,
        "next_gap_flags_t": next_gap_flags_t,
        "rel_gap_to_next_t": rel_gap_to_next_t,
        "any_5pct_flag_pollution_per_cluster": any_5pct_pollution,
        "any_5pct_flag_socio_per_cluster": any_5pct_socio,
        "five_pct_flag_both_types_per_cluster": both_types,
    }

# run of the function
path =r'...'
for batch_nb in [1,2,3]:
    df_full = pd.read_csv(path+'\\'+'AQ_census_2021_EU_PCA_clusters_batch%d.csv'%batch_nb)
    countries=np.unique(df_full['Country'].values)
    pollutant_cols = ["MEAN_pm_25", "MEAN_pm_10", "MEAN_no_2", "MEAN_o3_p932"]
    socio_cols = ["MEAN_Y_LT15_share", "MEAN_Y_GE65_share",
        "MEAN_NAT_share", "MEAN_EU_OTH_share", "MEAN_OTH_share", "MEAN_EMP_share" ]
    
    df_full_ind=pd.DataFrame({})
    for country in countries:
        df=df_full[df_full['Country']==country].copy()
        results = flag_above_median_next_gap_v2(
            df=df,
            cluster_col="CLUSTER_ID",
            pollutant_cols=pollutant_cols,
            socio_cols=socio_cols,
            value_cols=pollutant_cols + socio_cols,
            pct_threshold=0.05 #
        )
        
        # Accessing outputs
        data_t = results["data_t"]
        medians = results["medians"].T
        above_flags = results["above_median_flags_t"].T
        
        flags = results["next_gap_flags_t"].T
        flags.columns=[pollutant_cols[i]+'_f' for i in np.arange(0,len(pollutant_cols))] + [socio_cols[i]+'_f' for i in np.arange(0,len(socio_cols))] 
      
        gaps  = results["rel_gap_to_next_t"].T
        gaps.columns=[pollutant_cols[i]+'_g' for i in np.arange(0,len(pollutant_cols))] + [socio_cols[i]+'_g' for i in np.arange(0,len(socio_cols))]
        any_poll = results["any_5pct_flag_pollution_per_cluster"].T
        any_socio = results["any_5pct_flag_socio_per_cluster"].T
        both_types = results["five_pct_flag_both_types_per_cluster"].T
        
        df.index= np.arange(0,len(df))
        above_flags.index=np.arange(0,len(above_flags))
        df_full_ind_temp=pd.concat([df,flags.astype(int).reset_index(drop=True)],axis=1)
        df_full_ind_temp=pd.concat([df_full_ind_temp,gaps.reset_index(drop=True)],axis=1)
        df_full_ind_temp=pd.concat([df_full_ind_temp,any_poll.astype(int).to_frame().reset_index(drop=True)],axis=1)
        df_full_ind_temp=pd.concat([df_full_ind_temp,any_socio.astype(int).to_frame().reset_index(drop=True)],axis=1)
        df_full_ind_temp=pd.concat([df_full_ind_temp,both_types.astype(int).to_frame().reset_index(drop=True)],axis=1)
        df_full_ind=pd.concat([df_full_ind,df_full_ind_temp],axis=0)
        
    df_full_ind=df_full_ind.reset_index(drop=True)
    df_full_ind['hotspot']=df_full_ind[[pollutant_cols[i]+'_f' for i in np.arange(0,len(pollutant_cols))]].any(axis=1).astype(int)
    df_full_ind['hotspot_elderly']=df_full_ind['hotspot'] & df_full_ind['MEAN_Y_GE65_share_f']
    df_full_ind['hotspot_children']=df_full_ind['hotspot'] & df_full_ind['MEAN_Y_LT15_share_f']
    df_full_ind['hotspot_employment']=df_full_ind['hotspot'] & df_full_ind['MEAN_EMP_share_f']
    df_full_ind['hotspot_natives']=df_full_ind['hotspot'] & df_full_ind['MEAN_NAT_share_f']
    df_full_ind['hotspot_foreigners']=df_full_ind['hotspot'] & (df_full_ind[['MEAN_OTH_share_f','MEAN_EU_OTH_share_f']].any(axis=1).astype(int))
    df_full_ind.to_csv(path+'\\'+"AQ_census_2021_EU_PCA_clusters_batch%d_with_indicators.csv"%(batch_nb), index=False)



