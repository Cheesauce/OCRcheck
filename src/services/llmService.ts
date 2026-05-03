// Service for integrating with the Gemini API to provide AI features
// like OCR text auto-correction.

const apiKey = ""; 

async function fetchWithBackoff(url: string, options: RequestInit, retries = 5) {
  const delays = [1000, 2000, 4000, 8000, 16000];
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const err = new Error(`HTTP ${response.status}: ${errorData?.error?.message || 'Unknown error'}`);
        (err as any).status = response.status;
        throw err;
      }
      return await response.json();
    } catch (error: any) {
      // Fail fast on client errors like 400, 403, or 404 so we can immediately try the next model
      if (i === retries - 1 || [400, 403, 404].includes(error.status)) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, delays[i]));
    }
  }
}

export async function autoCorrectOcrText(rawText: string): Promise<string> {
  if (!rawText.trim()) return rawText;

  // Enhanced prompt to explicitly instruct the AI to use pattern recognition 
  // across phrases and sentences to determine the correct words.
  const systemPrompt = `You are an advanced OCR correction AI. Your task is to fix spelling, typographical, and formatting errors in the provided OCR text.
  
Key Instructions:
1. Recognize patterns of words, common phrases, and sentence structures to infer the correct context.
2. If a word is heavily garbled by the OCR process, use the surrounding sentence pattern to determine what the intended word was and correct it.
3. Fix fragmented sentences, misplaced punctuation, and broken hyphenated words.
4. EXTREMELY IMPORTANT: DO NOT alter proper nouns, brand names, company names, model numbers, person names, or technical identifiers (e.g., KONICA MINOLTA, LAPTOP-JMOGT6MU). Assume capitalized distinctive words are correct unless they are obvious minor scanner artifacts of a known standard word.
5. Output ONLY the perfectly corrected text. Do NOT include conversational filler, markdown fences, or explanations.
6. Preserve the original meaning, tone, intent, and layout of the document.`;

  const userQuery = `Raw OCR text:\n${rawText}`;

  const payload = {
    contents: [{ parts: [{ text: userQuery }] }],
    systemInstruction: { parts: [{ text: systemPrompt }] },
    generationConfig: {
      temperature: 0.2, // Keep it focused and deterministic for accurate corrections
    }
  };

  // Failover list: The service will try these models in order until one succeeds
  const modelsToTry = [
    'gemini-2.5-flash-preview-09-2025'
  ];

  for (const model of modelsToTry) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    
    try {
      const result = await fetchWithBackoff(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const text = result?.candidates?.[0]?.content?.parts?.[0]?.text;
      
      // If we got a valid response, return it immediately
      if (text) {
        return text.trim();
      }
    } catch (err: any) {
      console.warn(`[AI Service] Model ${model} failed (${err.message}). Trying next fallback...`);
      // It will continue to the next model in the array
    }
  }

  // If all models in the failover list fail, return the raw uncorrected text
  console.error("[AI Service] All failover models failed. Returning raw OCR text.");
  return rawText;
}
