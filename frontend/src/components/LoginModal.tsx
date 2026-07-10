import React, { useState, useEffect } from 'react';
import { X, User, Heart, ArrowUp, Sun, Cloud, Mail, AlertCircle, CheckCircle } from 'lucide-react';
import { generateOTP, sendOTPEmail, verifyOTP } from '../services/emailService';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLogin: (userData: any) => void;
}

const LoginModal: React.FC<LoginModalProps> = ({ isOpen, onClose, onLogin }) => {
  const [step, setStep] = useState<'login' | 'otp' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const [name, setName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  const [otpTimer, setOtpTimer] = useState(0);
  const [canResendOTP, setCanResendOTP] = useState(false);

  // Timer for OTP resend
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (otpTimer > 0) {
      interval = setInterval(() => {
        setOtpTimer((prev) => {
          if (prev <= 1) {
            setCanResendOTP(true);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [otpTimer]);

  // Reset form when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      resetForm();
    }
  }, [isOpen]);

  const resetForm = () => {
    setStep('login');
    setEmail('');
    setOtp('');
    setName('');
    setErrors({});
    setOtpTimer(0);
    setCanResendOTP(false);
  };

  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleRequestOTP = async () => {
    const newErrors: {[key: string]: string} = {};
    
    if (!email) {
      newErrors.email = 'Email is required';
    } else if (!validateEmail(email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);
    setErrors({});
    
    try {
      const existingUsers = JSON.parse(localStorage.getItem('users') || '[]');
      const userExists = existingUsers.find((user: any) => user.email === email);
      
      const otp = generateOTP();
      const emailSent = await sendOTPEmail(email, otp);
      
      if (emailSent) {
        if (userExists) {
          setStep('otp');
        } else {
          setStep('register');
        }
        setOtpTimer(60);
        setCanResendOTP(false);
      } else {
        setErrors({ general: 'Failed to send OTP. Please try again.' });
      }
    } catch (error) {
      console.error('Error during authentication:', error);
      setErrors({ general: 'An error occurred. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendOTP = async () => {
    if (!canResendOTP) return;
    
    setIsLoading(true);
    try {
      const otp = generateOTP();
      const emailSent = await sendOTPEmail(email, otp);
      
      if (emailSent) {
        setOtpTimer(60);
        setCanResendOTP(false);
        setErrors({});
      } else {
        setErrors({ general: 'Failed to resend OTP. Please try again.' });
      }
    } catch (error) {
      console.error('Error resending OTP:', error);
      setErrors({ general: 'An error occurred. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyOTP = async () => {
    if (!otp) {
      setErrors({ otp: 'Please enter the OTP' });
      return;
    }
    
    setIsLoading(true);
    setErrors({});
    
    try {
      const isValid = verifyOTP(email, otp);
      
      if (isValid) {
        const existingUsers = JSON.parse(localStorage.getItem('users') || '[]');
        const user = existingUsers.find((u: any) => u.email === email);
        
        if (user) {
          onLogin(user);
          onClose();
        }
      } else {
        setErrors({ otp: 'Invalid OTP. Please check and try again.' });
      }
    } catch (error) {
      console.error('Error verifying OTP:', error);
      setErrors({ general: 'An error occurred. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async () => {
    const newErrors: {[key: string]: string} = {};
    
    if (!name.trim()) {
      newErrors.name = 'Name is required';
    }
    if (!email) {
      newErrors.email = 'Email is required';
    }
    if (!otp) {
      newErrors.otp = 'OTP is required';
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);
    setErrors({});
    
    try {
      const isValid = verifyOTP(email, otp);
      
      if (isValid) {
        const newUser = { 
          name: name.trim(), 
          email: email,
          id: Date.now(),
          createdAt: new Date().toISOString()
        };
        
        const existingUsers = JSON.parse(localStorage.getItem('users') || '[]');
        existingUsers.push(newUser);
        localStorage.setItem('users', JSON.stringify(existingUsers));
        
        onLogin(newUser);
        onClose();
      } else {
        setErrors({ otp: 'Invalid OTP. Please check and try again.' });
      }
    } catch (error) {
      console.error('Error during registration:', error);
      setErrors({ general: 'An error occurred. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    onClose();
    resetForm();
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-4xl w-full mx-auto overflow-hidden">
        <div className="flex min-h-[600px]">
          {/* Left Panel - Blue Background */}
          <div className="bg-gradient-to-br from-[#2874F0] to-[#1e5cb8] text-white p-8 w-1/2 relative hidden lg:block">
            <div className="h-full flex flex-col justify-between">
              <div>
                <h2 className="text-4xl font-bold mb-4">Welcome Back!</h2>
                <p className="text-xl opacity-90 leading-relaxed">
                  Get access to your Orders, Wishlist and Recommendations
                </p>
                <div className="mt-6 space-y-3">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-300" />
                    <span>Fast & Secure Checkout</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-300" />
                    <span>Exclusive Deals & Offers</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-300" />
                    <span>24/7 Customer Support</span>
                  </div>
                </div>
              </div>
              
              {/* Illustration */}
              <div className="relative">
                <div className="flex items-center justify-center space-x-4 mb-4">
                  <div className="bg-red-500 p-3 rounded-lg shadow-lg">
                    <Heart className="h-5 w-5 text-white" />
                  </div>
                  <div className="bg-gray-800 p-3 rounded-lg shadow-lg">
                    <ArrowUp className="h-5 w-5 text-white" />
                  </div>
                </div>
                
                <div className="flex items-center justify-center space-x-4 mb-4">
                  <div className="bg-white p-4 rounded-lg shadow-lg">
                    <User className="h-8 w-8 text-[#2874F0]" />
                  </div>
                  <div className="bg-yellow-400 p-3 rounded-lg shadow-lg">
                    <span className="text-blue-600 font-bold text-lg">SK</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-center space-x-3">
                  <Sun className="h-5 w-5 text-yellow-300" />
                  <Cloud className="h-5 w-5 text-gray-300" />
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - White Background */}
          <div className="bg-white p-8 w-full lg:w-1/2 relative">
            <button
              onClick={handleClose}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="h-6 w-6" />
            </button>

            {step === 'login' && (
              <div className="space-y-6">
                <div className="text-center lg:text-left">
                  <h2 className="text-3xl font-bold text-gray-900 mb-2">Login</h2>
                  <p className="text-gray-600">Welcome back to Synergy Kart</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Email Address
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full pl-10 pr-3 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2874F0] focus:border-transparent"
                      placeholder="Enter your email"
                    />
                  </div>
                  {errors.email && (
                    <p className="text-red-500 text-sm mt-1 flex items-center">
                      <AlertCircle className="h-4 w-4 mr-1" />
                      {errors.email}
                    </p>
                  )}
                </div>

                {errors.general && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                    <p className="text-red-600 text-sm flex items-center">
                      <AlertCircle className="h-4 w-4 mr-2" />
                      {errors.general}
                    </p>
                  </div>
                )}

                <button
                  onClick={handleRequestOTP}
                  disabled={isLoading}
                  className="w-full bg-[#FB641B] text-white py-3 px-4 rounded-lg font-medium hover:bg-[#e55a17] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Sending OTP...
                    </div>
                  ) : (
                    'Request OTP'
                  )}
                </button>

                <p className="text-xs text-gray-500 text-center">
                  By continuing, you agree to Synergy Kart's{' '}
                  <a href="#" className="text-[#2874F0] underline hover:text-[#1e5cb8]">Terms of Use</a> and{' '}
                  <a href="#" className="text-[#2874F0] underline hover:text-[#1e5cb8]">Privacy Policy</a>.
                </p>
              </div>
            )}

            {step === 'otp' && (
              <div className="space-y-6">
                <div className="text-center lg:text-left">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">Enter OTP</h3>
                  <p className="text-gray-600">
                    We've sent a 6-digit code to {email}
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Enter 6-digit OTP
                  </label>
                  <input
                    type="text"
                    value={otp}
                    onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2874F0] focus:border-transparent text-center text-2xl tracking-widest"
                    placeholder="000000"
                    maxLength={6}
                  />
                  {errors.otp && (
                    <p className="text-red-500 text-sm mt-1 flex items-center">
                      <AlertCircle className="h-4 w-4 mr-1" />
                      {errors.otp}
                    </p>
                  )}
                </div>

                {otpTimer > 0 && (
                  <p className="text-sm text-gray-600 text-center">
                    Resend OTP in {formatTime(otpTimer)}
                  </p>
                )}

                <button
                  onClick={handleResendOTP}
                  disabled={!canResendOTP || isLoading}
                  className="w-full text-[#2874F0] py-2 hover:underline disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Sending...' : 'Resend OTP'}
                </button>

                <button
                  onClick={handleVerifyOTP}
                  disabled={!otp || isLoading}
                  className="w-full bg-[#FB641B] text-white py-3 px-4 rounded-lg font-medium hover:bg-[#e55a17] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Verifying...
                    </div>
                  ) : (
                    'Verify OTP'
                  )}
                </button>

                <button
                  onClick={() => setStep('login')}
                  className="w-full text-[#2874F0] py-2 hover:underline"
                >
                  Back to Login
                </button>
              </div>
            )}

            {step === 'register' && (
              <div className="space-y-6">
                <div className="text-center lg:text-left">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">Create Account</h3>
                  <p className="text-gray-600">
                    Welcome! Please enter your details to create your account.
                  </p>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name
                    </label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2874F0] focus:border-transparent"
                      placeholder="Enter your full name"
                    />
                    {errors.name && (
                      <p className="text-red-500 text-sm mt-1 flex items-center">
                        <AlertCircle className="h-4 w-4 mr-1" />
                        {errors.name}
                      </p>
                    )}
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address
                    </label>
                    <input
                      type="email"
                      value={email}
                      disabled
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none bg-gray-50"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Enter OTP
                    </label>
                    <p className="text-xs text-gray-500 mb-2">
                      We've sent a 6-digit code to {email}
                    </p>
                    <input
                      type="text"
                      value={otp}
                      onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#2874F0] focus:border-transparent text-center text-2xl tracking-widest"
                      placeholder="000000"
                      maxLength={6}
                    />
                    {errors.otp && (
                      <p className="text-red-500 text-sm mt-1 flex items-center">
                        <AlertCircle className="h-4 w-4 mr-1" />
                        {errors.otp}
                      </p>
                    )}
                  </div>
                </div>

                {errors.general && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                    <p className="text-red-600 text-sm flex items-center">
                      <AlertCircle className="h-4 w-4 mr-2" />
                      {errors.general}
                    </p>
                  </div>
                )}

                <button
                  onClick={handleRegister}
                  disabled={!name || !otp || isLoading}
                  className="w-full bg-[#FB641B] text-white py-3 px-4 rounded-lg font-medium hover:bg-[#e55a17] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoading ? (
                    <div className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Creating Account...
                    </div>
                  ) : (
                    'Create Account'
                  )}
                </button>

                <button
                  onClick={() => setStep('login')}
                  className="w-full text-[#2874F0] py-2 hover:underline"
                >
                  Back to Login
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginModal; 