// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// function App() {
//   // State variables to store data
//   const [textToAdd, setTextToAdd] = useState('');
//   const [question, setQuestion] = useState('');
//   const [answer, setAnswer] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [timing, setTiming] = useState(null);
//   const [hasKnowledgeBase, setHasKnowledgeBase] = useState(false);

//   // Your FastAPI backend URL (change this to match your deployment)
//   const API_URL = 'http://localhost:8000';

//   // Function to add text to knowledge base
//   const addText = async () => {
//     if (!textToAdd.trim()) {
//       alert('Please enter some text first!');
//       return;
//     }

//     setLoading(true);
//     const startTime = Date.now();

//     try {
//       const endpoint = hasKnowledgeBase ? '/add-text' : '/create-knowledge-base';
//       const response = await axios.post(`${API_URL}${endpoint}`, {
//         text: textToAdd
//       });

//       const endTime = Date.now();
//       const timeTaken = (endTime - startTime) / 1000;

//       alert(`Text added successfully! (${timeTaken.toFixed(2)}s)`);
//       setTextToAdd('');
//       setHasKnowledgeBase(true);
//       setTiming(timeTaken);

//     } catch (error) {
//       alert('Error adding text: ' + error.response?.data?.detail || error.message);
//     }

//     setLoading(false);
//   };

//   // Function to ask a question
//   const askQuestion = async () => {
//     if (!question.trim()) {
//       alert('Please enter a question!');
//       return;
//     }

//     setLoading(true);
//     const startTime = Date.now();

//     try {
//       const response = await axios.post(`${API_URL}/query`, {
//         question: question
//       });

//       const endTime = Date.now();
//       const timeTaken = (endTime - startTime) / 1000;

//       setAnswer(response.data);
//       setTiming(timeTaken);

//     } catch (error) {
//       alert('Error asking question: ' + error.response?.data?.detail || error.message);
//     }

//     setLoading(false);
//   };

//   // Function to estimate costs (rough calculation)
//   const estimateTokens = (text) => {
//     // Rough estimate: 1 token ≈ 4 characters
//     return Math.ceil(text.length / 4);
//   };

//   const estimateCost = (tokens) => {
//     // Rough estimate: $0.000001 per token (adjust based on actual pricing)
//     return (tokens * 0.000001).toFixed(6);
//   };

//   return (
//     <div className="App">
//       <header className="App-header">
//         <h1>🤖 RAG Chat System</h1>
//         <p>Add text to create knowledge base, then ask questions!</p>
//       </header>

//       <main className="main-content">
//         {/* Section 1: Add Text */}
//         <section className="section">
//           <h2>📝 Add Text to Knowledge Base</h2>
//           <div className="input-group">
//             <textarea
//               value={textToAdd}
//               onChange={(e) => setTextToAdd(e.target.value)}
//               placeholder="Paste or type your text here..."
//               rows="6"
//               className="text-area"
//             />
//             <button 
//               onClick={addText} 
//               disabled={loading}
//               className="primary-button"
//             >
//               {loading ? 'Adding...' : (hasKnowledgeBase ? 'Add More Text' : 'Create Knowledge Base')}
//             </button>
//           </div>
          
//           {/* Show text stats */}
//           {textToAdd && (
//             <div className="stats">
//               <small>
//                 Characters: {textToAdd.length} | 
//                 Estimated tokens: ~{estimateTokens(textToAdd)} | 
//                 Estimated cost: ~${estimateCost(estimateTokens(textToAdd))}
//               </small>
//             </div>
//           )}
//         </section>

//         {/* Section 2: Ask Questions */}
//         <section className="section">
//           <h2>❓ Ask a Question</h2>
//           <div className="input-group">
//             <input
//               type="text"
//               value={question}
//               onChange={(e) => setQuestion(e.target.value)}
//               placeholder="What would you like to know?"
//               className="question-input"
//               onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
//             />
//             <button 
//               onClick={askQuestion} 
//               disabled={loading || !hasKnowledgeBase}
//               className="primary-button"
//             >
//               {loading ? 'Thinking...' : 'Ask Question'}
//             </button>
//           </div>
          
//           {!hasKnowledgeBase && (
//             <p className="warning">⚠️ Please add some text first!</p>
//           )}
//         </section>

//         {/* Section 3: Show Timing Info */}
//         {timing && (
//           <section className="timing-section">
//             <div className="timing-info">
//               ⏱️ Last request took: <strong>{timing.toFixed(2)} seconds</strong>
//             </div>
//           </section>
//         )}

//         {/* Section 4: Show Answer and Sources */}
//         {answer && (
//           <section className="section answer-section">
//             <h2>💬 Answer</h2>
            
//             <div className="answer-box">
//               <div className="answer-text">
//                 {answer.answer}
//               </div>
              
//               <div className="answer-meta">
//                 <small>
//                   Query processed at: {new Date(answer.timestamp).toLocaleTimeString()} | 
//                   Sources used: {answer.num_sources}
//                   {question && (
//                     <span> | Estimated tokens: ~{estimateTokens(question + answer.answer)} | 
//                     Estimated cost: ~${estimateCost(estimateTokens(question + answer.answer))}</span>
//                   )}
//                 </small>
//               </div>
//             </div>

//             {/* Show Sources */}
//             {answer.sources && answer.sources.length > 0 && (
//               <div className="sources-section">
//                 <h3>📚 Sources & Citations</h3>
//                 {answer.sources.map((source, index) => (
//                   <div key={index} className="source-item">
//                     <div className="source-header">
//                       <strong>Source {source.index}:</strong>
//                     </div>
//                     <div className="source-content">
//                       {source.content}
//                     </div>
//                   </div>
//                 ))}
//               </div>
//             )}
//           </section>
//         )}
//       </main>
//     </div>
//   );
// }

// export default App;

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // State variables to store data
  const [textToAdd, setTextToAdd] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [timing, setTiming] = useState(null);
  const [hasKnowledgeBase, setHasKnowledgeBase] = useState(false);
  const [uploadedTexts, setUploadedTexts] = useState([]); // NEW: Store uploaded texts

  // Your FastAPI backend URL (change this to match your deployment)
  const API_URL = 'http://localhost:8000';

  // ADD THIS SNIPPET HERE:
  // Auto-scroll to bottom when answer is generated
  useEffect(() => {
    if (answer) {
      scrollToBottom();
    }
  }, [answer]);

  // Function to add text to knowledge base
  const addText = async () => {
    if (!textToAdd.trim()) {
      alert('Please enter some text first!');
      return;
    }

    setLoading(true);
    const startTime = Date.now();

    try {
      const endpoint = hasKnowledgeBase ? '/add-text' : '/create-knowledge-base';
      const response = await axios.post(`${API_URL}${endpoint}`, {
        text: textToAdd
      });

      const endTime = Date.now();
      const timeTaken = (endTime - startTime) / 1000;

      alert(`Text added successfully! (${timeTaken.toFixed(2)}s)`);
      
      // NEW: Add uploaded text to display list
      setUploadedTexts(prev => [...prev, {
        id: Date.now(),
        text: textToAdd,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
      setTextToAdd('');
      setHasKnowledgeBase(true);
      setTiming(timeTaken);

    } catch (error) {
      alert('Error adding text: ' + error.response?.data?.detail || error.message);
    }

    setLoading(false);
  };

  // Function to ask a question
  const askQuestion = async () => {
    if (!question.trim()) {
      alert('Please enter a question!');
      return;
    }

    setLoading(true);
    const startTime = Date.now();

    try {
      const response = await axios.post(`${API_URL}/query`, {
        question: question
      });

      const endTime = Date.now();
      const timeTaken = (endTime - startTime) / 1000;

      setAnswer(response.data);
      setTiming(timeTaken);

    } catch (error) {
      alert('Error asking question: ' + error.response?.data?.detail || error.message);
    }

    setLoading(false);
  };

  // Function to estimate costs (rough calculation)
  const estimateTokens = (text) => {
    // Rough estimate: 1 token ≈ 4 characters
    return Math.ceil(text.length / 4);
  };

  const estimateCost = (tokens) => {
    // Rough estimate: $0.000001 per token (adjust based on actual pricing)
    return (tokens * 0.000001).toFixed(6);
  };

  // Function to scroll to bottom
  const scrollToBottom = () => {
    window.scrollTo({
    top: document.documentElement.scrollHeight,
    behavior: 'smooth'
  });
};

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Chat System</h1>
        <p>Add text to create knowledge base, then ask questions!</p>
        <p>NOTICE: Use the <b>Scroll-Down</b> button to see the ANSWER/OUTPUT.</p>
      </header>

      <main className="main-content">
        {/* Section 1: Add Text */}
        <section className="section">
          <h2>Add Text to Knowledge Base</h2>
          <div className="input-group">
            <textarea
              value={textToAdd}
              onChange={(e) => setTextToAdd(e.target.value)}
              placeholder="Paste or type your text here..."
              rows="6"
              className="text-area"
            />
            <button 
              onClick={addText} 
              disabled={loading}
              className="primary-button"
            >
              {loading ? 'Adding...' : (hasKnowledgeBase ? 'Add More Text' : 'Create Knowledge Base')}
            </button>
          </div>
          
          {/* Show text stats */}
          {textToAdd && (
            <div className="stats">
              <small>
                Characters: {textToAdd.length} | 
                Estimated tokens: ~{estimateTokens(textToAdd)} | 
                Estimated cost: ~${estimateCost(estimateTokens(textToAdd))}
              </small>
            </div>
          )}

          {/* NEW: Display uploaded texts */}
          {uploadedTexts.length > 0 && (
            <div className="uploaded-texts">
              <h3>Uploaded Knowledge Base Content:</h3>
              {uploadedTexts.map((item) => (
                <div key={item.id} className="uploaded-text-item">
                  <div className="uploaded-text-meta">
                    <small>Added at: {item.timestamp}</small>
                  </div>
                  <div className="uploaded-text-content">
                    {item.text}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Section 2: Ask Questions */}
        <section className="section">
          <h2>Ask a Question</h2>
          <div className="input-group">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What would you like to know?"
              className="question-input"
              onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
            />
            <button 
              onClick={askQuestion} 
              disabled={loading || !hasKnowledgeBase}
              className="primary-button"
            >
              {loading ? 'Thinking...' : 'Ask Question'}
            </button>
          </div>
          
          {!hasKnowledgeBase && (
            <p className="warning">Please add some text first!</p>
          )}
        </section>

        {/* Section 3: Show Timing Info */}
        {timing && (
          <section className="timing-section">
            <div className="timing-info">
              Last request took: <strong>{timing.toFixed(2)} seconds</strong>
            </div>
          </section>
        )}

        {/* Section 4: Show Answer and Sources */}
        {answer && (
          <section className="section answer-section">
            <h2>Answer</h2>
            
            <div className="answer-box">
              <div className="answer-text">
                {answer.answer}
              </div>
              
              <div className="answer-meta">
                <small>
                  Query processed at: {new Date(answer.timestamp).toLocaleTimeString()} | 
                  Sources used: {answer.num_sources}
                  {question && (
                    <span> | Estimated tokens: ~{estimateTokens(question + answer.answer)} | 
                    Estimated cost: ~${estimateCost(estimateTokens(question + answer.answer))}</span>
                  )}
                </small>
              </div>
            </div>

            {/* Show Sources */}
            {answer.sources && answer.sources.length > 0 && (
              <div className="sources-section">
                <h3>Sources & Citations</h3>
                {answer.sources.map((source, index) => (
                  <div key={index} className="source-item">
                    <div className="source-header">
                      <strong>Source {source.index}:</strong>
                    </div>
                    <div className="source-content">
                      {source.content}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}
      </main>
      {/* Scroll to bottom button */}
      <button 
        onClick={scrollToBottom} 
        className="scroll-down-button"
        title="Scroll to bottom"
      >
        ↓
      </button>
    </div>
  );
}


export default App;
