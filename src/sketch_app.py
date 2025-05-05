import os
from PIL import Image
import streamlit as st
from streamlist_drawable_canvas import st_canvas
from src.retrieve_icons import IconRetriever

st.set_page_config(page_title="Icone Sketch Retrieval", layout="wide")

st.title("🖌️ Icon Sketch-to-Icon Retrieval Demo")
st.markdown("Draw a simple sketch on the left, and see matching icons on the right.")\

st.sidebar.header("Options")
k = st.sidebar.slider("Number of matches (k)", 1, 10, 5)

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=5,
    stroke_color="#000000",
    background_color="#FFFFFF", 
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_reslt.image_data is not None:
    sketch = Image.fromarray(canvas_result.image_data.astype("uint8"))
    st.sidebar.image(sketch, caption="Your Sketch", width=128)

    ir = IconRetriever(device="cpu")
    matches = ir.retrieve(sketch, k=k)

    st.subheader("Top Matches")
    cols = st.columns(min(k, 5))
    for idx, icon_name in enumerate(matches):
        with cols[idx % len(cols)]:
            icon_path = os.path.join("data", "icons", icon_name)
            st.image(icon_path, caption=icon_name, width=64)