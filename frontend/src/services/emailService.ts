import emailjs from '@emailjs/browser';

// Initialize EmailJS
emailjs.init('v8TJW4Y89kQ0frrXp');

// EmailJS Configuration
// You'll need to sign up at https://www.emailjs.com/ and get these IDs
const EMAILJS_CONFIG = {
  SERVICE_ID: 'service_4tce5p8', // Replace with your EmailJS service ID
  TEMPLATE_ID: 'template_x7e4xir', // Replace with your EmailJS template ID
  PUBLIC_KEY: 'v8TJW4Y89kQ0frrXp', // Replace with your EmailJS public key
};

// Generate a random 6-digit OTP
export const generateOTP = (): string => {
  return Math.floor(100000 + Math.random() * 900000).toString();
};

// Store OTPs temporarily (in production, use a proper database)
const otpStore = new Map<string, { otp: string; timestamp: number }>();

// Send OTP via EmailJS
export const sendOTPEmail = async (email: string, otp: string): Promise<boolean> => {
  try {
    console.log('--- DEVELOPMENT MODE ---');
    console.log(`Mocking Email sent to: ${email}`);
    console.log(`Your Synergy Kart OTP is: ${otp}`);
    console.log('------------------------');
    
    // Store OTP with timestamp (valid for 10 minutes)
    otpStore.set(email, {
      otp,
      timestamp: Date.now()
    });

    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Auto-alert the OTP so the user can easily see it without checking the console
    alert(`[DEV MODE] Your OTP is: ${otp}`);

    return true;
  } catch (error: any) {
    console.error('Error sending email:', error);
    return false;
  }
};

// Verify OTP
export const verifyOTP = (email: string, inputOTP: string): boolean => {
  const storedData = otpStore.get(email);
  
  if (!storedData) {
    return false;
  }

  // Check if OTP is expired (10 minutes)
  const isExpired = Date.now() - storedData.timestamp > 10 * 60 * 1000;
  
  if (isExpired) {
    otpStore.delete(email);
    return false;
  }

  // Check if OTP matches
  if (storedData.otp === inputOTP) {
    otpStore.delete(email); // Remove OTP after successful verification
    return true;
  }

  return false;
};

// Clean up expired OTPs
export const cleanupExpiredOTPs = () => {
  const now = Date.now();
  for (const [email, data] of otpStore.entries()) {
    if (now - data.timestamp > 10 * 60 * 1000) {
      otpStore.delete(email);
    }
  }
};

// Run cleanup every 5 minutes
setInterval(cleanupExpiredOTPs, 5 * 60 * 1000);

// Test function to verify EmailJS setup
export const testEmailJS = async (): Promise<boolean> => {
  try {
    console.log('Testing EmailJS configuration...');
    console.log('Service ID:', EMAILJS_CONFIG.SERVICE_ID);
    console.log('Template ID:', EMAILJS_CONFIG.TEMPLATE_ID);
    console.log('Public Key:', EMAILJS_CONFIG.PUBLIC_KEY);
    
    const testParams = {
      to_email: 'test@example.com',
      otp: '123456',
      message: 'Test email from Synergy Kart'
    };
    
    const response = await emailjs.send(
      EMAILJS_CONFIG.SERVICE_ID,
      EMAILJS_CONFIG.TEMPLATE_ID,
      testParams,
      EMAILJS_CONFIG.PUBLIC_KEY
    );
    
    console.log('Test email sent successfully:', response);
    return true;
  } catch (error: any) {
    console.error('Test email failed:', error);
    console.error('Error details:', {
      message: error?.message,
      status: error?.status,
      text: error?.text
    });
    return false;
  }
}; 