import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { User, Menu, X, LogOut, ChevronDown } from 'lucide-react';
import SearchBar from './SearchBar';
import CartIcon from './CartIcon';
import { useUser } from '../context/UserContext';
import LoginModal from './LoginModal';

const Header: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isUserDropdownOpen, setIsUserDropdownOpen] = useState(false);
  const { user, login, logout } = useUser();

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const handleLogin = (userData: any) => {
    login(userData);
  };

  return (
    <>
      <header className="bg-[#2874F0] shadow-md sticky top-0 z-50">
        <div className="max-w-8xl mx-auto px-6 sm:px-8 lg:px-12">
          <div className="flex items-center justify-between h-20">
            {/* Logo */}
            <div className="flex items-center">
              <Link to="/" className="flex items-center space-x-4">
                <div className="flex items-center">
                  <img 
                    src="/logo.jpg" 
                    alt="Synergy Kart Logo" 
                    className="h-12 w-auto object-contain"
                  />
                </div>
                <div className="text-white">
                  <div className="font-bold text-2xl">SYNERGY KART</div>
                  <div className="text-sm text-[#FFEA00]">Explore Plus</div>
                </div>
              </Link>
            </div>

            {/* Search Bar - Desktop */}
            <div className="hidden md:flex flex-1 max-w-4xl mx-16">
              <SearchBar />
            </div>

            {/* User Actions */}
            <div className="flex items-center space-x-10">
              {user ? (
                <div className="hidden md:block relative">
                  <div 
                    className="flex items-center space-x-2 text-white cursor-pointer hover:text-[#FFD700] transition-colors"
                    onMouseEnter={() => setIsUserDropdownOpen(true)}
                    onMouseLeave={() => setIsUserDropdownOpen(false)}
                  >
                    <User className="h-6 w-6" />
                    <span className="text-lg font-medium">{user.name}</span>
                    <ChevronDown className={`h-4 w-4 transition-transform ${isUserDropdownOpen ? 'rotate-180' : ''}`} />
                  </div>
                  
                  {/* Dropdown Menu */}
                  {isUserDropdownOpen && (
                    <div 
                      className="absolute top-full right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-50"
                      onMouseEnter={() => setIsUserDropdownOpen(true)}
                      onMouseLeave={() => setIsUserDropdownOpen(false)}
                    >
                      <div className="px-4 py-2 border-b border-gray-100">
                        <p className="text-sm text-gray-600">Signed in as</p>
                        <p className="font-medium text-gray-900">{user.name}</p>
                      </div>
                      <button 
                        onClick={logout}
                        className="w-full flex items-center space-x-3 px-4 py-2 text-left text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        <LogOut className="h-4 w-4" />
                        <span className="text-sm">Logout</span>
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <button 
                  onClick={() => setIsLoginModalOpen(true)}
                  className="hidden md:flex items-center space-x-3 text-white hover:text-[#FFD700] transition-colors"
                >
                  <User className="h-7 w-7" />
                  <span className="text-lg font-medium">Login</span>
                </button>
              )}
              <CartIcon />
              <button
                className="md:hidden text-white hover:text-[#FFD700] transition-colors"
                onClick={toggleMobileMenu}
              >
                {isMobileMenuOpen ? <X className="h-8 w-8" /> : <Menu className="h-8 w-8" />}
              </button>
            </div>
          </div>

          {/* Mobile Search Bar */}
          <div className="md:hidden pb-6">
            <SearchBar />
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden bg-[#1e5cb8] border-t border-blue-400">
            <nav className="px-4 py-4 space-y-2">
              <Link
                to="/"
                className="block text-white hover:text-[#FFD700] py-2 transition-colors"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Home
              </Link>
              {user ? (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2 text-white py-2">
                    <User className="h-5 w-5" />
                    <span className="font-medium">{user.name}</span>
                  </div>
                  <div className="pl-7 space-y-1">
                    <button 
                      onClick={logout}
                      className="flex items-center space-x-2 text-white hover:text-[#FFD700] py-2 transition-colors"
                    >
                      <LogOut className="h-4 w-4" />
                      <span className="text-sm">Logout</span>
                    </button>
                  </div>
                </div>
              ) : (
                <button 
                  onClick={() => setIsLoginModalOpen(true)}
                  className="flex items-center space-x-2 text-white hover:text-[#FFD700] py-2 transition-colors"
                >
                  <User className="h-5 w-5" />
                  <span>Login</span>
                </button>
              )}
            </nav>
          </div>
        )}
      </header>
      
      {/* Login Modal */}
      <LoginModal
        isOpen={isLoginModalOpen}
        onClose={() => setIsLoginModalOpen(false)}
        onLogin={handleLogin}
      />
    </>
  );
};

export default Header;