import streamlit as st
import pandas as pd
import time 
import ast
import plotly.graph_objects as go
from PIL import Image


@st.cache(suppress_st_warning=True)
def load_data():
    filebase = 'https://s3-eu-west-1.amazonaws.com/files.innerdoc.com/christmas-emotion-detection/'

    df_songs = pd.read_csv(filebase+'songs.csv', sep=';', header=0, encoding='utf-8')
    df_emotions = pd.read_csv(filebase+'plutchik.csv', sep=';', header=0, encoding='utf-8')
    df_labels = pd.read_csv(filebase+'songlabels.csv', sep=';', header=0, encoding='utf-8')

    return df_songs, df_emotions, df_labels


def create_fig():
    # Create figure
    fig3 = go.Figure()

    # Add images
    try:
        img = Image.open('christmas-lyrics-emotion-detector/plutchik_model_of_emotions_with_faces.png')
    except:
        img = Image.open('plutchik_model_of_emotions_with_faces.png')
    fig3.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=-16,
                y=15.8,
                sizex=32,
                sizey=32,
                sizing="contain", # contain stretch
                opacity=1,
                layer="below")
    )

    # Set template and margin
    fig3.update_layout(template="plotly_white",
        margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0  #top margin
            )
    )

    # Set axes properties # , nticks=64)#
    fig3.update_xaxes(range=[-16, 16], zeroline=False, showline=False, showgrid=False, showticklabels=False)
    fig3.update_yaxes(range=[-16, 16], zeroline=False, showline=False, showgrid=False, showticklabels=False)

    # Set figure size
    fig3.update_layout(width=600, height=600)

    return fig3


def highlight(s):
    return [f'background-color: {s["Emotion Color"]};']*len(s)


def highlight_column(color):
    return f'background-color: {color};'


def emphasize_column(color):
    return f'font-weight:bold;'



st.sidebar.title('Christmas Lyrics Emotion Detector')
st.sidebar.markdown(f"---")

# load data
df_songs, df_emotions, df_labels = load_data()


# prepare data
songs = df_songs.Title.values.tolist()
song = st.sidebar.selectbox('Select Christmas Song', songs, index=0)
df_song = df_songs[df_songs['Title'] == song]
song_id = df_song['Id'].values[0]
artist = df_song['Artist'].values[0]
title = df_song['Title'].values[0]
album = df_song['Album'].values[0]
youtube_url = df_song['Youtube_url'].values[0]
released = df_song['Released'].values[0]
df_songlabels = df_labels[df_labels['song_id'] == song_id]
sentence_id = st.sidebar.slider('Slide to Lyric-sentence', 0, len(df_songlabels), max(0,14))


# sidebar
st.sidebar.markdown(f"---")
show_about = st.sidebar.checkbox('About this App')
if show_about:
    st.subheader('About')
    st.markdown('''This *Christmas Lyrics Emotion Detector* is created by [Rob van Zoest](https://www.linkedin.com/in/robvanzoest/) and started with the idea for a fun pre-christmas project: \n\n \
- The **code** is available on [github.com/innerdoc](https://github.com/innerdoc/christmas-lyrics-emotion-detector)\n \
- The **emotions** were taken from [Plutchik's wheel of Emotions](https://en.wikipedia.org/wiki/Robert_Plutchik)\n \
- The **data** was generated with the help of *Zero-shot emotion classification* with [Transformers from Huggingface](https://joeddav.github.io/blog/2020/05/29/ZSL.html)\n \
and this [Colab Notebook](https://colab.research.google.com/drive/1gcJq-6YXGca7i_8XUfq9z08MU_EWlkSd)\n \
- The **interface** is build with the help of [Streamlit](https://www.streamlit.io/) and [Plotly](https://plotly.com/python/)\n \
\n\nYou can follow me on [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40innerdoc)](https://twitter.com/innerdoc_nlp)
\n\n\n\n
---
\n\n\n\n
''')
st.sidebar.markdown(f"---")
st.sidebar.video(youtube_url)

st.sidebar.markdown(f"Artist : **{artist}**")
st.sidebar.markdown(f"Title : **{title}**")
st.sidebar.markdown(f"Year  : **{released}**")
st.sidebar.markdown(f"Album : **{album}**")



# lyric-sentence
placeholder_sent_meta = st.empty()
placeholder_sent = st.empty()

# plutchiks wheel
placeholder_fig = st.empty()
fig3 = create_fig()
placeholder_fig.plotly_chart(fig3)

with st.beta_expander('The Development of an Emotion', expanded=True):
    placeholder_emotion_table = st.empty()
    st.markdown('\n\n')

with st.beta_expander('Lyrics and Labels', expanded=True):
    st.markdown(f"<span style='font-size:12px;'>Labels are made of the *Primary Emotion* with the *Effect* of that emotion.</span>", unsafe_allow_html=True)
    placeholder_lyrics = st.empty()

lyrics_text = []
for index, line in df_songlabels.iterrows():
    if line.sent_id == sentence_id:
        fig3 = create_fig()
    labels_tags = ''
    labels = ast.literal_eval(line.labels)

    # emotion labels
    for idx, l in enumerate(labels):
        df_emo = df_emotions[df_emotions['id'] == l[0]] # UNK !
        prim_label = df_emo.primary_emotion.values[0]
        bgcolor = df_emo.primary_emotion_color.values[0]
        bcolor = df_emo.intense_emotion_color.values[0]
        effect = df_emo.effect.values[0]

        labels_tags += f"<span style='color:white; background-color:{bgcolor}; border-color:{bcolor}; border-style:solid; border-width: 3px; border-radius: 5%; margin-left: 10px; padding: 2px 6px;'>{prim_label} > {effect}</span> "

        if line.sent_id == sentence_id:
            stimulus_event = df_emo.stimulus_event.values[0]
            cognitive_appraisal = df_emo.cognitive_appraisal.values[0]
            behavior = df_emo.behavior.values[0]
            function = df_emo.function.values[0]

            if prim_label == 'fear':
                x0, y0 = 6.2, 0
            if prim_label == 'surprise':
                x0, y0 = 4.4, -4.4
            if prim_label == 'sadness':
                x0, y0 = 0, -6.4
            if prim_label == 'disgust':
                x0, y0 = -4.4, -4.4
            if prim_label == 'anger':
                x0, y0 = 4.4, 4.4
            if prim_label == 'anticipation':
                x0, y0 = -4.4, 4.4
            if prim_label == 'joy':
                x0, y0 = 0, 6.5
            if prim_label == 'trust':
                x0, y0 = 4.4, 4.4
            
            if prim_label:
                r=3
                shape_props = {"type":"circle", "xref":"x", "yref":"y", "opacity":.5, "fillcolor":"black", "line_color":"black"}
                fig3.add_shape(x0=x0-r, y0=y0-r, x1=x0+r, y1=y0+r, **shape_props)
                r=.2
                shape_props = {"type":"circle", "xref":"x", "yref":"y", "opacity":.4, "fillcolor":"red", "line_color":"red", "line_width":3 }
                fig3.add_shape(x0=x0-r, y0=y0-r, x1=x0+r, y1=y0+r, **shape_props)

    if line.sent_id == sentence_id:
        placeholder_sent.markdown(f"<span style='font-size:28px; font-weight:bold; font-style:italic'>\"{line.sent}\"</span>\n\n{labels_tags}", unsafe_allow_html=True)
        placeholder_sent_meta.markdown(f"<span style='font-size:12px;'>Emotions for the {released}-song **{title}** by **{artist}**, sentence **{line.sent_id}**: </span>", unsafe_allow_html=True)

        placeholder_fig.plotly_chart(fig3)
        
        # emotion table
        try:
            label_ids = [x[0] for x in labels if x[0] != -1]
            subset_labels = df_emotions[df_emotions['id'].isin(label_ids)][['stimulus_event','cognitive_appraisal','primary_emotion','behavior','function','effect','primary_emotion_color']]
            subset_labels.columns = ['Stimulus Event','Cognitive Appraisal','Primary Emotion','Behavior','Function','Effect','Emotion Color']
            placeholder_emotion_table.table(subset_labels.style.applymap(highlight_column, subset=['Emotion Color']).applymap(emphasize_column, subset=['Primary Emotion']) ) # .apply(highlight, axis=1)
        except:
            pass

        sent_info = f"**{line.sent_id}  -  {line.sent}** {labels_tags}"
    else:
        sent_info = f"{line.sent_id}  -  {line.sent} {labels_tags}"
    lyrics_text.append(sent_info)
        
placeholder_lyrics.markdown('\n\n'.join(lyrics_text), unsafe_allow_html=True)

# social
st.markdown('[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40innerdoc)](https://twitter.com/innerdoc_nlp)')





debug = '''
with st.beta_expander('Emotions'):
    st.table(df_emotions)

with st.beta_expander('Songs'):
    st.dataframe(df_songs)

with st.beta_expander('Song Labels'):
    st.table(df_labels)
'''