import pandas as pd
import plotly.express as px
import streamlit as st


def render_charts(summary):
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.markdown("##### Publications by year")
        if summary["years"]:
            df_years = pd.DataFrame(list(summary["years"].items()), columns=["Year", "Articles"])
            fig_years = px.bar(
                df_years, x="Year", y="Articles",
                color_discrete_sequence=["#002B49"],
                template="plotly_white"
            )
            fig_years.update_layout(margin=dict(l=20, r=20, t=30, b=20), font_family="Inter, sans-serif")
            st.plotly_chart(fig_years, width="stretch")
        else:
            st.caption("No temporal data available.")
            
    with c_chart2:
        st.markdown("##### Screening decisions")
        if summary["decisions"]:
            df_dec = pd.DataFrame(list(summary["decisions"].items()), columns=["Decision", "Count"])
            fig_pie = px.pie(
                df_dec,
                names="Decision",
                values="Count",
                color="Decision",
                color_discrete_map={"Include": "#10B981", "Exclude": "#EF4444", "Borderline": "#F59E0B", "Unscreened": "#94A3B8"},
                template="plotly_white",
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20), font_family="Inter, sans-serif")
            st.plotly_chart(fig_pie, width="stretch")
            
    c_chart3, c_chart4 = st.columns(2)
    with c_chart3:
        st.markdown("##### Leading journals")
        if summary["top_journals"]:
            df_j = pd.DataFrame(list(summary["top_journals"].items())[:10], columns=["Journal", "Count"])
            fig_j = px.bar(
                df_j, y="Journal", x="Count", orientation="h",
                color_discrete_sequence=["#EB6500"],
                template="plotly_white"
            )
            fig_j.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=30, b=20), font_family="Inter, sans-serif")
            st.plotly_chart(fig_j, width="stretch")
            
    with c_chart4:
        st.markdown("##### Keyword frequency")
        if summary["top_keywords"]:
            df_kw = pd.DataFrame(list(summary["top_keywords"].items())[:12], columns=["Keyword", "Frequency"])
            fig_kw = px.bar(
                df_kw, y="Keyword", x="Frequency", orientation="h",
                color_discrete_sequence=["#007398"],
                template="plotly_white"
            )
            fig_kw.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=30, b=20), font_family="Inter, sans-serif")
            st.plotly_chart(fig_kw, width="stretch")
