import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import CategoryNav from './components/CategoryNav';
import Homepage from './components/Homepage';
import SearchResults from './components/SearchResults';
import TestSearch from './components/TestSearch';
import CartPage from './components/CartPage';
import CheckoutPage from './components/CheckoutPage';
import Footer from './components/Footer';
import { SearchProvider } from './context/SearchContext';
import { UserProvider } from './context/UserContext';
import { CartProvider } from './context/CartContext';

function App() {
  return (
    <UserProvider>
      <SearchProvider>
        <CartProvider>
          <Router>
            <div className="min-h-screen bg-gray-50">
              <Header />
              <CategoryNav />
              <main>
                <Routes>
                  <Route path="/" element={<Homepage />} />
                  <Route path="/search" element={<SearchResults />} />
                  <Route path="/test" element={<TestSearch />} />
                  <Route path="/cart" element={<CartPage />} />
                  <Route path="/checkout" element={<CheckoutPage />} />
                </Routes>
              </main>
              <Footer />
            </div>
          </Router>
        </CartProvider>
      </SearchProvider>
    </UserProvider>
  );
}

export default App;