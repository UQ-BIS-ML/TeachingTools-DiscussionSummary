
# **Discussion Summariser**

**Discussion Summariser** is a web application built with **Gradio** that uses **OpenAI's GPT-4** models to generate concise and engaging summaries of discussion posts. This tool is ideal for educators who want to provide students with clear recaps of class discussions.

---

## **Features**
- 📄 **Upload Discussion Files:** Supports `.xlsx` files with discussion data.
- 🧠 **Model Selection:** Choose between different OpenAI models (e.g., `gpt-4o-mini`, `gpt-4o`).
- 🎯 **Contextual Summaries:** Generate summaries with a customizable context.
- ✍️ **Word Count Control:** Set a word limit for the generated summary.
- 💾 **Save Output:** Summaries are saved as `.txt` files in the `Summaries` folder.

---

## **Installation**

### 1. **Clone the Repository:**
```bash
git clone https://github.com/UQ-BIS-ML/TeachingTools-DiscussionSummary.git
cd discussion-summariser
```

### 2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

### 3. **Set Up Environment Variables:**
Create a `.env` file and add your **OpenAI API Key**:
```env
PERSONAL_OPENAI_KEY=your_api_key_here
```

### 4. **Prepare Directories:**
Ensure the following directories exist:
```bash
mkdir -p Discussions Summaries
```

---

## **Usage**

### **Run the Application:**
```bash
python discussion_summarizer.py
```

### **Open the Interface:**
Navigate to `http://localhost:7860` in your browser.

### **Generate Summaries:**
- Upload an **Excel file** containing discussion data.
- Select the **OpenAI model**.
- Set the **word count limit**.
- Edit the **context** as needed.
- Click **Generate** to produce the summary.

---

## **File Structure**

```bash
.
├── Discussions     # Directory for input Excel files
├── Summaries              # Directory where summaries are saved
├── discussion_summarizer.py                 # Main application file
├── requirements.txt       # Python dependencies
└── .env                   # API key configuration
```

---

## **Dependencies**

- **OpenAI**: For language model interaction.
- **Gradio**: To build the web interface.
- **Pandas**: To handle Excel files.
- **Tenacity**: For retry logic in API calls.
- **dotenv**: To manage environment variables.

Install them via:

```bash
pip install openai gradio pandas tenacity python-dotenv
```

---

## **Troubleshooting**

- 🛠 **API Key Issues:** Double-check the `.env` file.
- 🛠 **File Upload Errors:** Ensure `.xlsx` files are formatted correctly.
- 🛠 **Model Errors:** Verify the `AVALIABLE_MODELS` list matches available OpenAI models.

---

## **Contributing**

Feel free to submit **pull requests** or **issues** to improve the project!

---

## **License**

This project is licensed under the **MIT License**.
