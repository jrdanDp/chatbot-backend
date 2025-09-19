from langchain_ollama.llms import OllamaLLM  # ✅ cambio de import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager  # ✅ necesario para streaming real
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # ✅ muestra tokens
from app.rag_loader import retriever
import uuid
from typing import Generator

# Diccionarios para mantener el estado por sesión
session_memories = {}
session_names = {}
session_greeted = set()

def stream_chatbot(user_input: str, session_id: str = None) -> Generator[str, None, None]:
    if session_id is None:
        session_id = str(uuid.uuid4())

    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            ai_prefix="Asistente",
            human_prefix="Usuario",
            memory_key="chat_history",
            return_messages=True
        )

    memory = session_memories[session_id]

    if session_id not in session_names:
        if any(word in user_input.lower() for word in ["soy", "me llamo", "mi nombre es"]):
            posibles = user_input.split()
            for i, palabra in enumerate(posibles):
                if palabra.lower() in ["soy", "llamo", "es"] and i + 1 < len(posibles):
                    session_names[session_id] = posibles[i + 1].capitalize()
                    break
        else:
            yield "Hola, ¿cómo te llamas? Me gustaría saber tu nombre para poder acompañarte mejor."
            return

    if session_id not in session_greeted:
        nombre = session_names.get(session_id, "")
        session_greeted.add(session_id)
        yield f"¡Hola {nombre}! Me alegra hablar contigo. ¿Cómo te sientes hoy?"
        return

    prompt = PromptTemplate.from_template(
        """
    Eres un compañero emocional digital que conversa en **español** como un amigo cercano.

    🧠 Estilo:
    - Responde con frases breves (2 a 5 líneas como máximo).
    - Usa emojis suaves (❤️ ✨) solo cuando sea natural.
    - Mantén un tono cálido, afectuoso y cercano.
    - No repitas tu presentación ni digas que eres una IA.

    🎯 Enfoque de respuestas (proporción 70/30):
    - 70% del tiempo: Da consejos simples, recomendaciones prácticas o ideas que puedan ayudar.
    - 30% del tiempo: Valida emociones o profundiza con preguntas abiertas.

    📌 Prioriza este flujo:
    1. Valida brevemente la emoción (si aplica).
    2. Ofrece un consejo o apoyo práctico de forma clara y afectuosa.
    3. Solo si es oportuno, añade una pregunta breve para invitar a compartir más.

    Ejemplos:
    - Consejo breve → "Podrías probar escribir lo que sientes. A veces ayuda ❤️"
    - Sugerencia práctica → "Salir a caminar unos minutos puede ayudarte a despejar la mente."
    - Validación → "Siento que estés pasando por esto... ¿Qué te ayudaría ahora?"
    - Estímulo → "¡Qué bien! ¿Qué fue lo que más te gustó? ✨"

    Historial de conversación:
    {chat_history}

    Información recuperada:
    {context}

    Pregunta del usuario:
    {question}
    """
    )



    # ✅ LLM con streaming real usando CallbackManager
    llm = OllamaLLM(
        model="mistral",
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    # Conversational RAG
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    # Asegurar que siempre se yield algo
    try:
        response_stream = qa_chain.stream({"question": user_input})
        for chunk in response_stream:
            content = chunk.get("answer", "")
            if content:
                yield content
    except Exception as e:
        yield f"Lo siento, ocurrió un error al procesar tu mensaje: {str(e)}"