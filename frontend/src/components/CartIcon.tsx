import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useCart } from '../context/CartContext';

const CartIcon: React.FC = () => {
  const { state } = useCart();
  const navigate = useNavigate();

  return (
    <button
      onClick={() => navigate('/cart')}
      className="relative flex items-center space-x-2 text-white hover:text-[#FFD700] transition-colors"
    >
      {/* Custom Cart Icon similar to Flipkart */}
      <div className="relative">
        <svg 
          className="h-8 w-8" 
          viewBox="0 0 24 24" 
          fill="currentColor"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M7 18c-1.1 0-1.99.9-1.99 2S5.9 22 7 22s2-.9 2-2-.9-2-2-2zM1 2v2h2l3.6 7.59-1.35 2.45c-.16.28-.25.61-.25.96 0 1.1.9 2 2 2h12v-2H7.42c-.14 0-.25-.11-.25-.25l.03-.12L8.1 13h7.45c.75 0 1.41-.41 1.75-1.03L21.7 4H5.21l-.94-2H1zm16 16c-1.1 0-1.99.9-1.99 2s.89 2 1.99 2 2-.9 2-2-.9-2-2-2z"/>
        </svg>
        {state.itemCount > 0 && (
          <span className="absolute -top-2 -right-2 bg-[#FFD700] text-[#2874F0] text-xs rounded-full h-6 w-6 flex items-center justify-center font-bold">
            {state.itemCount > 99 ? '99+' : state.itemCount}
          </span>
        )}
      </div>
      <span className="text-lg font-medium">Cart</span>
    </button>
  );
};

export default CartIcon; 