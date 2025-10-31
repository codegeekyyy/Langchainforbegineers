from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text="""🧠 Step 2 — Make sure you’re using the right Python

Activate your virtual environment before running:

cd "C:\Users\harsh\OneDrive\Desktop\Langchain-Models"
venv\Scripts\activate


Then reinstall inside that environment:

pip install -U langchain-ollama


Now check again:

pip show langchain-ollama

✅ Step 3 — Confirm in Python

Run this in the same environment:

python
>>> import langchain_ollama
>>> print(langchain_ollama.__file__)


If that works and prints a path, you’re good.

⚙️ Step 4 — Run your script again
python webpageloader.py


If it still fails after this, please copy–paste the outputs of:

pip show langchain-ollama
pip list | findstr langchain
where python


That will tell us exactly which Python environment your script is using, and I’ll pinpoint the fix for you instantly.

Would you like me to guide you through a quick one-liner that ensures langchain-ollama is installed in the same environment your VS Code (or terminal) uses?

You said:
give me a random 500 length text
ChatGPT said:

Sure 👍 — here’s a random 500-character text (exactly 500 characters long):

The evening sky shimmered with fading hues of violet and amber as a cool breeze swept through the quiet streets. Lanterns flickered one by one, painting soft circles of light upon the cobblestones. Somewhere nearby, a clock tower chimed, its echo weaving gently through the whispers of night. People moved unhurriedly, their footsteps blending with distant laughter and music. The city felt alive yet calm, breathing slowly as if savoring the final, peaceful moments before midnight’s arrival.

You said:
from langchain.text_splitter import RecursiveCharacterTextSplitter



splitter=RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
)
res=splitter.split_text()
print(res)
ChatGPT said:

Good attempt 👍 — you’re almost there!

The small issue is that split_text() requires an argument — you need to pass in the text you want to split.

❌ Your line (causes error)
res = splitter.split_text()

✅ Correct version
res = splitter.split_text(text)

✅ Full working code
from langchain.text_splitter import RecursiveCharacterTextSplitter



splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
)

res = splitter.split_text(text)
print(res)
"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=300,
    chunk_overlap=0
)

chunks=splitter.split_text(text)
print(chunks[0])
