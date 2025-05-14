import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fontainebleau Boulders", layout="wide")
st.title("Fontainebleau Boulders")
# Compact layout: reduce Streamlit default padding and vertical spacing
st.markdown("""
<style>
.main .block-container {
    padding-top: 0;
    padding-bottom: 0.5rem;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.25rem;
}
div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.1rem;
}
h1 {
    margin-top: 0;
    margin-bottom: 0;
}
h2, h3, h4, h5, h6 {
    margin-top: 0.2em;
    margin-bottom: 0.2em;
}
/* Hide anchor links next to headers (robust for all Streamlit versions) */
h1 > a[href^='#'], h2 > a[href^='#'], h3 > a[href^='#'], h4 > a[href^='#'], h5 > a[href^='#'], h6 > a[href^='#'],
.stMarkdown a[href^='#'],
.stMarkdown .header-anchor, .stMarkdown .css-1v0mbdj, .stMarkdown .css-1n76uvr,
.stMarkdown .stAnchor, .stMarkdown .stMarkdownAnchor, .stMarkdown .css-1aehpvj, .stMarkdown .css-1b7of8t {
    display: none !important;
}
/* Hide any anchor inside headings, regardless of class */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Inject CSS to limit multiselect dropdown height and enable scrolling
st.markdown("""
    <style>
    /* Limit height and enable scroll for multiselect dropdown options */
    div[data-baseweb="popover"] ul {
        max-height: 200px !important;
        overflow-y: auto !important;
    }
    /* Make expanders more compact */
    .stExpanderHeader {
        font-size: 1em;
        padding: 0.1em 0.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Load data robustly, handle parsing errors
def load_data():
    try:
        df = pd.read_csv("bleau_detailed_boulders.csv")
        return df, None
    except pd.errors.ParserError as e:
        try:
            df = pd.read_csv("bleau_detailed_boulders.csv", engine="python", on_bad_lines="warn")
            return df, None
        except Exception as e2:
            return None, f"Failed to load CSV: {e2}"

df, load_warning = load_data()

# Add Grade Group column
import re
def extract_grade_group(grade):
    if pd.isna(grade):
        return None
    m = re.match(r"(\d+)", str(grade))
    return m.group(1) if m else None
df["grade_group"] = df["grade"].apply(extract_grade_group)

if load_warning:
    st.warning(load_warning)
if df is None:
    st.error("Could not load the CSV file. Please check the file for formatting issues.")
    st.stop()

# Top-of-page filters, always visible
with st.container():
    # Easy-Access button logic
    EASY_ACCESS_SECTORS = [
        "Rocher Canon",
        "Rocher Canon Ouest",
        "Le Calvaire",
        "Roche d'Hercule",
        "Mont Ussy",
        "Mont Ussy Est",
    ]
    if 'easy_access' not in st.session_state:
        st.session_state['easy_access'] = False
    # Custom styled button with gray copy
    easy_access_label = "Easy Access Only"
    easy_access_desc = "Includes Rocher Canon, Rocher Canon Ouest, Le Calvaire, Roche d'Hercule, Mont Ussy, Mont Ussy Est"
    # Custom CSS for filled button when active
    st.markdown("""
    <style>
    .easy-access-row {display: flex; align-items: center; gap: 1em; margin-bottom: 0.5em;}
    .easy-access-desc {color: #888; font-size: 0.95em;}
    .stButton>button.easy-access-active {
        background: #2563eb !important;
        color: white !important;
        border: 1px solid #2563eb !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Use Streamlit-native button with strong CSS for width and nowrap
    # The button is filled (primary) when Easy Access is ON, outlined (secondary) when OFF
    st.markdown("""
        <style>
        div[data-testid="column"] button[data-testid^="baseButton-easy_access_btn"] {
            min-width: 280px !important;
            width: 100% !important;
            white-space: nowrap !important;
            font-size: 1rem !important;
            padding-left: 2em !important;
            padding-right: 2em !important;
        }
        .easy-access-helper-below {
            color: #666;
            font-size: 1em;
            margin-top: 0.1em;
            margin-bottom: 0.7em;
        }
        </style>
    """, unsafe_allow_html=True)
    # Always show all possible options for each filter
    all_sectors = sorted(df['sector'].dropna().unique().tolist())
    all_grade_groups = sorted(df['grade_group'].dropna().unique().tolist())
    all_grades = sorted(df['grade'].dropna().unique().tolist())
    # Apply easy-access filter if active
    sectors_to_show = EASY_ACCESS_SECTORS if st.session_state['easy_access'] else all_sectors

    # New row: Sectors and Easy Access side by side
    col_sectors, col_easy = st.columns([3, 9])
    with col_sectors:
        with st.expander("Sectors", expanded=True):
            selected_sector = st.selectbox("Select sector", ["All"] + sectors_to_show, index=0, key="sector_selectbox", label_visibility='collapsed')
    with col_easy:
        # Easy Access button and helper text in the same column
        st.write("")
        clicked = st.button(
            'Easy Access',
            key='easy_access_btn',
            type='primary' if st.session_state.get('easy_access', False) else 'secondary',
            help=None
        )
        if clicked:
            st.session_state['easy_access'] = not st.session_state.get('easy_access', False)
            st.rerun()
        st.markdown(
            '<div style="margin-top: 0; color: #666; font-size: 1em;">'
            'Filter for Rocher Canon, Rocher Canon Ouest, Le Calvaire, Roche d\'Hercule, Mont Ussy, Mont Ussy Est'
            '</div>',
            unsafe_allow_html=True
        )

    # Defaults
    all_sectors = sorted(df['sector'].dropna().unique().tolist())
    all_grade_groups = sorted(df['grade_group'].dropna().unique().tolist())
    all_grades = sorted(df['grade'].dropna().unique().tolist())

    # Session state for remembering selections
    if 'selected_sector' not in st.session_state:
        st.session_state['selected_sector'] = 'All'
    if 'selected_grade_groups' not in st.session_state:
        st.session_state['selected_grade_groups'] = all_grade_groups.copy()
    if 'selected_grades' not in st.session_state:
        st.session_state['selected_grades'] = all_grades.copy()

    # Always show all possible options for each filter
    all_sectors = sorted(df['sector'].dropna().unique().tolist())
    all_grade_groups = sorted(df['grade_group'].dropna().unique().tolist())
    all_grades = sorted(df['grade'].dropna().unique().tolist())
    # Apply easy-access filter if active
    sectors_to_show = EASY_ACCESS_SECTORS if st.session_state['easy_access'] else all_sectors

    # Start a new row of columns for Grades, Height, Stars, Ratings filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        with st.expander("Grades", expanded=True):
            import re
            def grade_sort_key(g):
                m = re.match(r"(\d+)([+-]?)", str(g))
                if not m:
                    return (999, 1, g)
                num = int(m.group(1))
                suf = m.group(2)
                suf_order = {'-': 0, '': 1, '+': 2}
                return (num, suf_order.get(suf, 3), str(g))
            import numpy as np
            def not_none_nan(x):
                return x is not None and not (isinstance(x, float) and np.isnan(x))
            all_grades = sorted([g for g in df['grade'].unique().tolist() if not_none_nan(g)], key=grade_sort_key)
            sorted_grades = [g for g in all_grades if not_none_nan(g)]
            if sorted_grades:
                idx_min, idx_max = 0, len(sorted_grades) - 1
                grade_indices = list(range(len(sorted_grades)))
                grade_labels = {i: sorted_grades[i] for i in grade_indices}
                grade_slider_label = f"Select grade range ({sorted_grades[0]} to {sorted_grades[-1]})"
                def robust_find_first_grade_idx(grade_list, prefix):
                    for i, g in enumerate(grade_list):
                        if not_none_nan(g) and str(g).startswith(prefix):
                            return i
                    return 0
                def robust_find_last_grade_idx(grade_list, prefix):
                    idx = -1
                    for i, g in enumerate(grade_list):
                        if not_none_nan(g) and str(g).startswith(prefix):
                            idx = i
                    return idx if idx != -1 else len(grade_list) - 1
                default_low_idx = robust_find_first_grade_idx(sorted_grades, "5")
                default_high_idx = robust_find_last_grade_idx(sorted_grades, "7")
                if default_low_idx is None or not isinstance(default_low_idx, int):
                    default_low_idx = idx_min
                if default_high_idx is None or not isinstance(default_high_idx, int):
                    default_high_idx = idx_max
                default_low_idx = max(idx_min, min(default_low_idx, idx_max))
                default_high_idx = max(idx_min, min(default_high_idx, idx_max))
                selected_idx_range = st.slider(
                    grade_slider_label,
                    min_value=idx_min,
                    max_value=idx_max,
                    value=(default_low_idx, default_high_idx),
                    step=1,
                    format="",
                    key="grade_slider",
                    label_visibility='collapsed',
                )
                selected_grades = sorted_grades[selected_idx_range[0]:selected_idx_range[1]+1]
                st.caption(f"Selected: {selected_grades[0]} to {selected_grades[-1]}")
            else:
                selected_grades = []
    with filter_col2:
        with st.expander("Stars", expanded=True):
            stars = df['stars'].dropna().astype(float)
            if not stars.empty:
                min_stars = float(np.floor(stars.min()))
                max_stars = float(np.ceil(stars.max()))
                default_stars = (3.5, max_stars) if 3.5 >= min_stars and 3.5 <= max_stars else (min_stars, max_stars)
                # Ensure both are float
                default_stars = (float(default_stars[0]), float(default_stars[1]))
                selected_stars = st.slider(
                    "Select stars range",
                    min_value=min_stars,
                    max_value=max_stars,
                    value=default_stars,
                    step=0.5,
                    format="%.1f",
                    key="stars_slider",
                    label_visibility='collapsed',
                )
            else:
                selected_stars = None
    with filter_col3:
        with st.expander("# Ratings", expanded=True):
            ratings = df['num_ratings'].dropna().astype(int)
            if not ratings.empty:
                min_ratings = int(ratings.min())
                max_ratings = int(ratings.max())
                selected_ratings = st.slider(
                    "Number of Ratings",
                    min_value=min_ratings,
                    max_value=max_ratings,
                    value=(10, max_ratings) if 10 >= min_ratings and 10 <= max_ratings else (min_ratings, max_ratings),
                    step=1,
                    key="ratings_slider",
                    label_visibility='collapsed',
                )
            else:
                selected_ratings = None
    with filter_col4:
        with st.expander("Shortest Climber Height", expanded=True):
            height_bands = sorted([str(x) for x in df['shortest_climber_height'].dropna().unique()])
            if height_bands:
                idx_min, idx_max = 0, len(height_bands) - 1
                selected_idx_range = st.slider(
                    "Select height band range",
                    min_value=idx_min,
                    max_value=idx_max,
                    value=(idx_min, idx_max),
                    step=1,
                    format="",
                    key="height_slider",
                    label_visibility='collapsed',
                )
                selected_height_bands = height_bands[selected_idx_range[0]:selected_idx_range[1]+1]
                st.caption(f"Selected: {selected_height_bands[0]} cm to {selected_height_bands[-1]} cm")
            else:
                selected_height_bands = []

# Use only the user's selections for filtering
filtered_df = df.copy()
# Apply Easy-Access sector filter if active
if 'easy_access' in st.session_state and st.session_state['easy_access']:
    filtered_df = filtered_df[filtered_df['sector'].isin(EASY_ACCESS_SECTORS)]
if selected_sector != "All":
    filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
# For grades, use the string-based slider range from sorted_grades
if 'sorted_grades' in locals() and selected_grades:
    filtered_df = filtered_df[filtered_df['grade'].isin(selected_grades)]
# For shortest climber height
if 'selected_height' in locals() and selected_height is not None:
    def parse_height(val):
        if pd.isna(val) or val == '':
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        m = re.match(r"(\d+)", str(val))
        return float(m.group(1)) if m else np.nan
    filtered_df = filtered_df[filtered_df['shortest_climber_height'].apply(parse_height).between(selected_height[0], selected_height[1], inclusive='both')]
# For stars
if 'selected_stars' in locals() and selected_stars is not None:
    filtered_df = filtered_df[(filtered_df['stars'].astype(float) >= selected_stars[0]) & (filtered_df['stars'].astype(float) <= selected_stars[1])]
# For # ratings
if 'selected_ratings' in locals() and selected_ratings is not None:
    filtered_df = filtered_df[(filtered_df['num_ratings'].astype(int) >= selected_ratings[0]) & (filtered_df['num_ratings'].astype(int) <= selected_ratings[1])]

# Define custom sort key for grades (used in filter)
def grade_sort_key(g):
    import re
    m = re.match(r"(\d+)([+-]?)", str(g))
    if not m:
        return (999, 1, g)  # Put non-matching grades at end
    num = int(m.group(1))
    suf = m.group(2)
    suf_order = {'-': 0, '': 1, '+': 2}
    return (num, suf_order.get(suf, 3), str(g))

# Show 'Number of Problems by Grade Group' and 'Grade' charts side-by-side (flipped)
col_group, col_grade = st.columns(2)
# --- Consistent color mapping for grades and grade groups ---
from matplotlib import cm
import matplotlib.colors as mcolors
import numpy as np
# 1. Assign each grade a unique color along the gradient
all_grades = filtered_df['grade'].dropna().unique().tolist()
all_grades = sorted(all_grades, key=grade_sort_key)
n_grades = len(all_grades)
grad_color_map = cm.get_cmap('RdYlGn_r')
grad_colors = [mcolors.to_hex(grad_color_map(i/(max(n_grades-1,1)))) for i in range(n_grades)]
grade_color_map = {str(g): c for g, c in zip(all_grades, grad_colors)}

# 2. For grade groups, average the RGB of their component grades
filtered_df['grade_group_label'] = filtered_df['grade_group'].fillna('Other').astype(str)
grade_group_to_grades = filtered_df.groupby('grade_group_label')['grade'].apply(lambda x: [str(g) for g in x.dropna()]).to_dict()
grade_group_color_map = {}
for group, grades in grade_group_to_grades.items():
    if group == 'Other' or not grades:
        grade_group_color_map[group] = '#888888'
    else:
        rgb_list = [mcolors.to_rgb(grade_color_map[g]) for g in grades if g in grade_color_map]
        if rgb_list:
            avg_rgb = tuple(np.mean(rgb_list, axis=0))
            grade_group_color_map[group] = mcolors.to_hex(avg_rgb)
        else:
            grade_group_color_map[group] = '#888888'
# Ensure all possible grade groups in charts are mapped
all_grade_groups = filtered_df['grade_group_label'].unique().tolist()
for g in all_grade_groups:
    if g not in grade_group_color_map:
        grade_group_color_map[g] = '#888888'

with col_group:
    # Replace None/null grade_group with 'Other'
    group_counts = filtered_df['grade_group'].fillna('Other').value_counts()
    # Move 'Other' to the end
    if 'Other' in group_counts.index:
        group_counts = group_counts[[g for g in group_counts.index if g != 'Other'] + ['Other']]
    # Convert to int and sort (excluding 'Other')
    try:
        idx_numeric = [int(float(x)) for x in group_counts.index if x != 'Other']
    except Exception:
        idx_numeric = [x for x in group_counts.index if x != 'Other']
    sorted_idx = sorted(idx_numeric)
    sorted_idx = [str(x) for x in sorted_idx] + (['Other'] if 'Other' in group_counts.index else [])
    group_counts = group_counts.loc[sorted_idx]
    x_labels = [str(x) for x in group_counts.index]
    fig_group = px.bar(
        x=x_labels,
        y=group_counts.values,
        color=x_labels,
        color_discrete_map=grade_group_color_map,
        labels={'x': 'Grade Group', 'y': 'Number of Problems'},
        title="Number of Problems by Grade Group"
    )
    fig_group.update_xaxes(type="category")
    fig_group.update_layout(
        showlegend=False,
        title_font_size=26  # 30% larger than typical 20px
    )
    fig_group.update_traces(hovertemplate='<b>Number of Problems:</b> %{y}<extra></extra>')
    st.plotly_chart(fig_group, use_container_width=True)
with col_grade:
    grade_counts = filtered_df['grade'].value_counts()
    grade_counts = grade_counts.loc[sorted(grade_counts.index, key=grade_sort_key)]
    grades_x = [str(g) for g in grade_counts.index]
    fig_grade = px.bar(
        x=grades_x,
        y=grade_counts.values,
        color=grades_x,
        color_discrete_map=grade_color_map,
        labels={'x': 'Grade', 'y': 'Number of Problems'},
        title="Number of Problems by Grade"
    )
    fig_grade.update_xaxes(type="category")
    fig_grade.update_layout(
        showlegend=False,
        title_font_size=26
    )
    fig_grade.update_traces(hovertemplate='<b>Number of Problems:</b> %{y}<extra></extra>')
    st.plotly_chart(fig_grade, use_container_width=True)

# Bar chart: Number of problems by sector (top 10)
if filtered_df['sector'].nunique() > 1:
    # Prepare data for stacked bar chart
    top_sectors = filtered_df['sector'].value_counts().nlargest(10).index.tolist()
    # Aggregate by sector and grade_label
    stacked_df = filtered_df[filtered_df['sector'].isin(top_sectors)].copy()
    stacked_df['grade_label'] = stacked_df['grade'].apply(lambda x: str(x) if pd.notna(x) else 'Other')
    summary_df = stacked_df.groupby(['sector', 'grade_label']).size().reset_index(name='count')
    # Use the same grade_color_map as above
    unique_grades = [g for g in grade_color_map.keys() if g in summary_df['grade_label'].unique()]
    import plotly.express as px
    fig_sector = px.bar(
        summary_df,
        x="sector",
        y="count",
        color="grade_label",
        category_orders={"sector": top_sectors, "grade_label": unique_grades},
        color_discrete_map=grade_color_map,
        labels={"sector": "Sector", "count": "Number of Problems", "grade_label": "Grade"},
        barmode="stack",
        title="Number of Problems by Sector (Top 10)",
        custom_data=["grade_label"]
    )
    fig_sector.update_layout(
        legend_title_text='Grade',
        hoverlabel=dict(align="left", bgcolor="white", bordercolor="black"),
        title_font_size=26
    )
    fig_sector.update_traces(
        hovertemplate='<b>Grade:</b> %{customdata[0]}<br><b>Number of Problems:</b> %{y}<extra></extra>'
    )
    st.plotly_chart(fig_sector, use_container_width=True)

# Bar chart: Average stars by grade
if 'stars' in filtered_df.columns:
    avg_stars = filtered_df.groupby('grade')['stars'].mean().dropna()
    # Sort index using grade_sort_key
    avg_stars = avg_stars.loc[sorted(avg_stars.index, key=grade_sort_key)]
    grades_x = [str(g) for g in avg_stars.index]
    # Use the same grade_color_map as above
    fig_stars = px.bar(
        x=grades_x,
        y=avg_stars.values,
        color=grades_x,
        color_discrete_map=grade_color_map,
        labels={'x': 'Grade', 'y': 'Average Stars'},
        title="Average Stars by Grade"
    )
    fig_stars.update_xaxes(type="category")
    fig_stars.update_layout(
        showlegend=False,
        title_font_size=26
    )
    fig_stars.update_traces(hovertemplate='<b>Average Stars:</b> %{y:.2f}<extra></extra>')
    st.plotly_chart(fig_stars, use_container_width=True)

# Data table of filtered problems

st.markdown('<div style="font-weight:600;font-size:1.56em;margin:0 0 0.1em 0;">List of Boulders</div>', unsafe_allow_html=True)

def format_grade(row):
    main = str(row['grade']) if pd.notna(row['grade']) else ''
    alt = str(row['alt_grade']) if 'alt_grade' in row and pd.notna(row['alt_grade']) and row['alt_grade'] != row['grade'] else ''
    return f"{main} ({alt})" if alt else main

def format_percent(val):
    try:
        return f"{round(float(val)/100*100)}%" if pd.notna(val) else ''
    except:
        return ''

# Prepare table columns in specified order
cols = [
    'name',
    'sector',
    'grade',
    'stars',
    'num_ratings',
    'most_recent',
    'shortest_climber_height',
    'grade_distribution',
    'perc_lower',
    'perc_higher',
    'tags',
    'url',
]
df_table = filtered_df.copy()
# Format grade
if 'grade' in df_table.columns and 'alt_grade' in df_table.columns:
    df_table['Grade'] = df_table.apply(format_grade, axis=1)
else:
    df_table['Grade'] = df_table['grade']
# Format perc_lower and perc_higher
if 'perc_lower' in df_table.columns:
    df_table['% soft'] = df_table['perc_lower'].apply(format_percent)
if 'perc_higher' in df_table.columns:
    df_table['% sandbagged'] = df_table['perc_higher'].apply(format_percent)
# Rename columns for display
col_map = {
    'name': 'Name',
    'sector': 'Sector',
    'Grade': 'Grade',
    'stars': 'Stars',
    'num_ratings': '# ratings',
    'most_recent': 'Most Recent',
    'shortest_climber_height': 'Shortest Climber',
    'grade_distribution': 'Grade Distribution',
    '% soft': '% soft',
    '% sandbagged': '% sandbagged',
    'tags': 'Tags',
    'url': 'URL',
}
display_cols = [
    'Name',
    'Sector',
    'Grade',
    'Stars',
    '# ratings',
    'Most Recent',
    'Shortest Climber',
    'Grade Distribution',
    '% soft',
    '% sandbagged',
    'Tags',
    'URL',
]
# Build display DataFrame
for k, v in col_map.items():
    if v not in df_table.columns and k in df_table.columns:
        df_table[v] = df_table[k]
# Ensure 'Grade' column is formatted and drop the original 'grade' column if present
if 'Grade' in df_table.columns:
    if 'grade' in df_table.columns:
        df_table = df_table.drop(columns=['grade'])
existing_cols = [col for col in display_cols if col in df_table.columns]
df_disp = df_table[existing_cols].copy()
# Ensure '# ratings' is integer (no decimals)
if '# ratings' in df_disp.columns:
    df_disp['# ratings'] = df_disp['# ratings'].apply(lambda x: int(x) if pd.notna(x) and x != '' else '')
# Replace None/NaN/null with blank cells
import numpy as np
df_disp = df_disp.replace([None, np.nan, float('nan')], '', regex=True)
from streamlit import dataframe, column_config

# Set custom column widths
col_cfg = {}
def_col_width = 150  # Default width in px (Streamlit default is ~150)
# No width set for 'Name' (default)
col_cfg['Sector'] = column_config.Column(width=int(def_col_width * 0.5))
col_cfg['Grade Distribution'] = column_config.Column(width=int(def_col_width * 0.72))

st.dataframe(df_disp, use_container_width=True, column_config=col_cfg)
