from langchain.text_splitter import RecursiveCharacterTextSplitter

text="""
The evening sky shimmered with fading hues of violet and amber as a cool breeze swept through the quiet streets. Lanterns flickered one by one, painting soft circles of light upon the cobblestones. Somewhere nearby, a clock tower chimed, its echo weaving gently through the whispers of night. People moved unhurriedly, their footsteps blending with distant laughter and music. The city felt alive yet calm, breathing slowly as if savoring the final, peaceful moments before midnight’s arrival.
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
)
res=splitter.split_text(text)
print(len(res))
print(res)