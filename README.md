**Description**

**Document Summarizer** is a Tkinter-based GUI application designed to streamline the process of document analysis. This application enables users to:

**Summarize Documents:** Using the advanced BART (Bidirectional and Auto-Regressive Transformers) model, the application generates concise summaries of lengthy documents, making it easier to grasp the main points quickly.

**Perform Sentiment Analysis**: The summarized text is analyzed to determine its sentiment, categorizing it as Beneficial, Risky, or Neutral. This feature is particularly useful for evaluating the tone and implications of contracts, reports, and other important documents.

**Text-to-Speech Functionality:** The application includes a text-to-speech engine that reads the summarized text aloud, offering an auditory option for consuming the information.


**Key Features**
**User-Friendly Interface:** The application is built using Tkinter, providing a simple and intuitive interface for users.

**Efficient Summarization**: Utilizes the BART model from the transformers library for high-quality text summarization.

**Sentiment Analysis**: Analyzes the sentiment of the text using TextBlob, helping users quickly understand the nature of the document.

**Text-to-Speech:** Converts text to speech using the pyttsx3 library, allowing users to listen to the summary.


**Software Requirements**
To run the Document Summarizer, you will need the following software:

**Python**: Ensure you have Python installed (version 3.6 or higher recommended).
**Tkinter**: A standard Python library for creating graphical user interfaces (comes pre-installed with Python).

Required Python Libraries

You can install the required libraries using pip:
**transformers**: For the BART model used in text summarization.
**pyttsx3**:For text-to-speech functionality.
**textblob**: For sentiment analysis.


