from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoTokenizer
from langdetect import detect
import streamlit as st


preferred_languages = ['en', 'de', 'fr', 'es', 'it', 'pt', 'ru', 'uk', 'hi']

langdetect_code_map = {
    'af': 'afr_Latn', 'am': 'amh_Ethi', 'ar': 'arb_Arab', 'az': 'azj_Latn', 'be': 'bel_Cyrl',
    'bn': 'ben_Beng', 'bs': 'bos_Latn', 'bg': 'bul_Cyrl', 'ca': 'cat_Latn', 'ceb': 'ceb_Latn',
    'cs': 'ces_Latn', 'cy': 'cym_Latn', 'da': 'dan_Latn', 'de': 'deu_Latn', 'el': 'ell_Grek',
    'en': 'eng_Latn', 'es': 'spa_Latn', 'et': 'est_Latn', 'fa': 'pes_Arab', 'fi': 'fin_Latn',
    'fr': 'fra_Latn', 'gu': 'guj_Gujr', 'he': 'heb_Hebr', 'hi': 'hin_Deva', 'hr': 'hrv_Latn',
    'hu': 'hun_Latn', 'id': 'ind_Latn', 'it': 'ita_Latn', 'ja': 'jpn_Jpan', 'jv': 'jav_Latn',
    'ka': 'kat_Geor', 'kk': 'kaz_Cyrl', 'kn': 'kan_Knda', 'ko': 'kor_Hang', 'lt': 'lit_Latn',
    'lv': 'lvs_Latn', 'ml': 'mal_Mlym', 'mr': 'mar_Deva', 'ms': 'zsm_Latn', 'my': 'mya_Mymr',
    'ne': 'npi_Deva', 'nl': 'nld_Latn', 'no': 'nob_Latn', 'or': 'ory_Orya', 'pa': 'pan_Guru',
    'pl': 'pol_Latn', 'pt': 'por_Latn', 'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'si': 'sin_Sinh',
    'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sq': 'als_Latn', 'sr': 'srp_Cyrl', 'sv': 'swe_Latn',
    'sw': 'swh_Latn', 'ta': 'tam_Taml', 'te': 'tel_Telu', 'th': 'tha_Thai', 'tr': 'tur_Latn',
    'uk': 'ukr_Cyrl', 'ur': 'urd_Arab', 'uz': 'uzn_Latn', 'vi': 'vie_Latn', 'zh-cn': 'zho_Hans',
    'zh-tw': 'zho_Hant'
}



def get_transcript(url):
    try:
        video_id = url.split('v=')[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)

        return ' '.join([i['text'] for i in transcript])
    except Exception as e:
        st.error(f"Transcript Error: {e}")
        return ""



def convert_lang(transcript):
    try:
        lang_code = detect(transcript)
        if lang_code in langdetect_code_map:
            return lang_code
        else:
            st.warning("Language detected but not supported.")
            return ""
    except Exception as e:
        st.error(f"Language detection failed: {e}")
        return ""



def split_into_chunks(text, tokenizer, max_tokens=700):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]



def generate_summary(transcript, lang_code, max_length=700):
    translate = pipeline("translation", model="facebook/nllb-200-distilled-600M")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    if lang_code != 'en':
        translated = translate(transcript, src_lang=langdetect_code_map[lang_code], tgt_lang=langdetect_code_map['en'])
        transcript = translated[0]['translation_text']

    chunks = split_into_chunks(transcript, tokenizer, max_tokens=700)
    summaries = summarizer(chunks, min_length=100, max_length=300)
    result = []

    return " ".join([s['summary_text'] for s in summaries])



st.set_page_config(page_title="Multilingual YouTube Summarizer", layout="centered")
st.title("🌐 YouTube Multilingual Video Summarizer")

url = st.text_input("Enter YouTube video URL:")

if url:
    if st.button("Summarize"):
        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(url)

        if transcript:
                st.spinner("Detecting language...")
                lang_code = convert_lang(transcript)
                if lang_code:
                    st.success(f"Detected language: {lang_code}")

                    with st.spinner("Generating summary..."):
                        final_summary = generate_summary(transcript, lang_code)
                        st.subheader("📄 Summary:")
                        st.write(final_summary)
                        st.success("✅ Done!")
                        st.download_button("📥 Download Summary as TXT", final_summary, file_name="summary.txt", mime="text/plain")
