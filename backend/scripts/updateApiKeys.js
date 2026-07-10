const fs = require('fs');
const path = require('path');

// Script to easily update multiple Gemini API keys
const envPath = path.join(__dirname, '..', '.env');

function updateApiKeys(apiKeys) {
  if (!Array.isArray(apiKeys) || apiKeys.length === 0) {
    console.error('❌ Please provide an array of API keys');
    return;
  }

  try {
    let envContent = fs.readFileSync(envPath, 'utf8');
    
    // Update each API key
    apiKeys.forEach((key, index) => {
      const keyNumber = index + 1;
      const keyPattern = new RegExp(`GEMINI_API_KEY_${keyNumber}=.*`, 'g');
      const newKeyLine = `GEMINI_API_KEY_${keyNumber}=${key}`;
      
      if (envContent.includes(`GEMINI_API_KEY_${keyNumber}=`)) {
        envContent = envContent.replace(keyPattern, newKeyLine);
      } else {
        // Add new key if it doesn't exist
        envContent += `\nGEMINI_API_KEY_${keyNumber}=${key}`;
      }
    });

    fs.writeFileSync(envPath, envContent);
    console.log('✅ API keys updated successfully!');
    console.log(`📊 Updated ${apiKeys.length} API keys`);
    
    // Show current keys (masked for security)
    apiKeys.forEach((key, index) => {
      const maskedKey = key.substring(0, 10) + '...' + key.substring(key.length - 4);
      console.log(`🔑 Key ${index + 1}: ${maskedKey}`);
    });
    
  } catch (error) {
    console.error('❌ Error updating API keys:', error);
  }
}

// Example usage - replace with your actual API keys
const yourApiKeys = [
  'AIzaSyB6i6TbnS0Q3ond6v5h49bi8m_ixavVZro', // Your current key
  'YOUR_SECOND_API_KEY_HERE',
  'YOUR_THIRD_API_KEY_HERE', 
  'YOUR_FOURTH_API_KEY_HERE'
];

// Uncomment the line below and add your actual API keys, then run: node scripts/updateApiKeys.js
// updateApiKeys(yourApiKeys);

console.log('📝 To use this script:');
console.log('1. Replace the API keys in the yourApiKeys array above');
console.log('2. Uncomment the updateApiKeys(yourApiKeys) line');
console.log('3. Run: node scripts/updateApiKeys.js');

module.exports = { updateApiKeys };
