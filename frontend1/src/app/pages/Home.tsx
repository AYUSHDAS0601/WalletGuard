import { useState } from "react";
import { useNavigate } from "react-router";
import { Shield, Search, AlertTriangle, TrendingUp, Activity, AlertCircle, CheckCircle, XCircle } from "lucide-react";
import { TrackerDetailsModal } from "../components/TrackerDetailsModal";

interface SuspiciousActivity {
  id: string;
  type: string;
  severity: "high" | "medium" | "low";
  description: string;
  timestamp: string;
  details: string;
}

// Mock data generator
const generateMockData = (address: string) => {
  const riskLevel = Math.random();
  
  let activities: SuspiciousActivity[] = [];
  let riskScore = 0;
  let status: "danger" | "warning" | "safe" = "safe";
  
  if (riskLevel > 0.6) {
    // High risk wallet
    status = "danger";
    riskScore = Math.floor(Math.random() * 25) + 75; // 75-100
    activities = [
      {
        id: "1",
        type: "High-Frequency Trading",
        severity: "high",
        description: "Detected 45 transactions within 10 minutes - possible bot activity",
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
        details: "This wallet executed 45 transactions in a 10-minute window, which is characteristic of automated bot trading. The transactions were made to multiple different addresses with varying amounts, suggesting potential wash trading or market manipulation activities.",
      },
      {
        id: "2",
        type: "Mixer Service Interaction",
        severity: "high",
        description: "Interacted with known mixing service",
        timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
        details: "The wallet has sent funds to a known cryptocurrency mixer (Tornado Cash). Mixers are used to obscure the origin of funds, which can be legitimate for privacy but is also commonly associated with money laundering activities.",
      },
      {
        id: "3",
        type: "Large Unusual Transfer",
        severity: "medium",
        description: "Transferred 150 ETH to unknown address",
        timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        details: "A transfer of 150 ETH was made to an address with no prior transaction history with this wallet. The receiving address was created less than 24 hours ago, which is a red flag for potential fund siphoning or preparation for exit scam.",
      },
    ];
  } else if (riskLevel > 0.3) {
    // Medium risk wallet
    status = "warning";
    riskScore = Math.floor(Math.random() * 35) + 40; // 40-75
    activities = [
      {
        id: "1",
        type: "Multiple Small Transactions",
        severity: "medium",
        description: "Unusual pattern of small transactions detected",
        timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
        details: "The wallet has been making numerous small-value transactions (under $10) to different addresses. This pattern is sometimes used in 'dusting attacks' or to test wallet security before larger fraudulent transactions.",
      },
      {
        id: "2",
        type: "New Address Interaction",
        severity: "low",
        description: "Frequent interaction with newly created addresses",
        timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
        details: "Over 60% of recent transactions involve addresses created within the last 30 days. While this alone isn't necessarily suspicious, it can indicate involvement in pump-and-dump schemes or new scam operations.",
      },
    ];
  } else {
    // Low risk wallet
    status = "safe";
    riskScore = Math.floor(Math.random() * 40); // 0-40
    activities = [
      {
        id: "1",
        type: "Normal Activity",
        severity: "low",
        description: "Regular transaction patterns detected",
        timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
        details: "This wallet exhibits normal transaction behavior consistent with regular cryptocurrency usage. Transaction frequency, amounts, and timing all fall within expected parameters for legitimate personal or business use.",
      },
    ];
  }

  return {
    address,
    status,
    riskScore,
    activities,
    balance: (Math.random() * 100).toFixed(4),
    transactionCount: Math.floor(Math.random() * 2000) + 500,
    firstSeen: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString(),
    lastActive: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  };
};

export function Home() {
  const [address, setAddress] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const [walletData, setWalletData] = useState<any>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (address.trim()) {
      setAnalyzing(true);
      setAnalyzed(false);
      
      // Simulate analysis
      setTimeout(() => {
        const data = generateMockData(address.trim());
        setWalletData(data);
        setAnalyzing(false);
        setAnalyzed(true);
      }, 2000);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "danger":
        return <XCircle className="text-red-600" size={32} />;
      case "warning":
        return <AlertTriangle className="text-amber-600" size={32} />;
      default:
        return <CheckCircle className="text-green-600" size={32} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "danger":
        return {
          bg: "bg-red-50",
          border: "border-red-200",
          text: "text-red-900",
          subtext: "text-red-700",
          badge: "bg-red-100 text-red-800 border-red-300",
        };
      case "warning":
        return {
          bg: "bg-amber-50",
          border: "border-amber-200",
          text: "text-amber-900",
          subtext: "text-amber-700",
          badge: "bg-amber-100 text-amber-800 border-amber-300",
        };
      default:
        return {
          bg: "bg-green-50",
          border: "border-green-200",
          text: "text-green-900",
          subtext: "text-green-700",
          badge: "bg-green-100 text-green-800 border-green-300",
        };
    }
  };

  const getStatusMessage = (status: string) => {
    switch (status) {
      case "danger":
        return {
          title: "⚠️ HIGH RISK - DANGER",
          message: "This wallet exhibits multiple high-risk behaviors. Avoid transacting with this address.",
        };
      case "warning":
        return {
          title: "⚡ MEDIUM RISK - WARNING",
          message: "Some suspicious patterns detected. Exercise caution when dealing with this wallet.",
        };
      default:
        return {
          title: "✓ LOW RISK - SAFE",
          message: "This wallet appears to have normal transaction patterns with minimal risk indicators.",
        };
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black relative overflow-hidden">
      {/* Animated Bubble */}
      {analyzed && walletData && (
        <div
          className={`fixed w-32 h-32 rounded-full blur-3xl transition-all duration-[2000ms] ease-out ${
            walletData.status === 'danger' ? 'bg-red-500' :
            walletData.status === 'warning' ? 'bg-amber-500' :
            'bg-green-500'
          }`}
          style={{
            bottom: analyzing ? '-10%' : '50%',
            right: analyzing ? '-10%' : '50%',
            transform: analyzing ? 'translate(0, 0) scale(0.5)' : 'translate(50%, 50%) scale(3)',
            opacity: analyzing ? 0 : 0.3,
            animation: 'bubble-appear 2s ease-out forwards',
          }}
        />
      )}

      {/* Header - Floating Navbar */}
      <header className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-2xl px-4">
        <nav className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-full shadow-2xl px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="text-blue-400" size={28} />
              <h1 className="text-xl font-bold text-white">WalletGuard</h1>
            </div>
            <div className="flex items-center gap-6">
              <a href="/" className="text-white hover:text-blue-400 transition-colors font-medium">
                Home
              </a>
              <button className="text-white/70 hover:text-blue-400 transition-colors font-medium">
                History
              </button>
              <button 
                onClick={() => analyzed && walletData && setIsModalOpen(true)}
                className="text-white/70 hover:text-blue-400 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!analyzed}
              >
                Know More
              </button>
            </div>
          </div>
        </nav>
      </header>

      {/* Hero Section */}
      <div className="max-w-5xl mx-auto px-4 py-16">
        <div className="text-center mb-12 mt-20">
          <h2 className="text-5xl font-bold text-white mb-4">
            Track Wallet Activity
          </h2>
          <p className="text-xl text-gray-300 mb-2">
            Monitor blockchain addresses for suspicious patterns and unusual behavior
          </p>
          <p className="text-sm text-gray-400">
            Enter any wallet address to analyze its transaction history
          </p>
        </div>

        {/* Search Form */}
        <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl shadow-xl p-8 mb-8">
          <form onSubmit={handleSubmit}>
            <div className="flex gap-4">
              <div className="flex-1">
                <input
                  type="text"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  placeholder="Enter wallet address (e.g., 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb)"
                  className="w-full px-6 py-4 text-lg bg-white/10 backdrop-blur-sm border-2 border-white/20 text-white placeholder-gray-400 rounded-xl focus:ring-2 focus:ring-blue-400 focus:border-blue-400 transition-all"
                  required
                />
              </div>
              <button
                type="submit"
                disabled={analyzing}
                className="px-8 py-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 transition-colors flex items-center gap-2 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Search size={20} />
                Track Wallet
              </button>
            </div>
          </form>
        </div>

        {/* Analysis Loading State */}
        {analyzing && (
          <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl shadow-xl p-12 mb-12">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-500/20 rounded-full mb-4 animate-pulse">
                <Search className="text-blue-400" size={32} />
              </div>
              <h3 className="text-2xl font-semibold text-white mb-2">Analyzing Wallet...</h3>
              <p className="text-gray-300">Scanning blockchain for suspicious activities</p>
              <div className="mt-6 flex justify-center">
                <div className="flex gap-2">
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analysis Results Section */}
        {analyzed && walletData && (
          <div className={`rounded-2xl shadow-xl p-8 mb-12 border-2 ${getStatusColor(walletData.status).bg} ${getStatusColor(walletData.status).border}`}>
            {/* Status Header */}
            <div className="flex items-start gap-4 mb-6">
              {getStatusIcon(walletData.status)}
              <div className="flex-1">
                <h3 className={`text-2xl font-bold mb-2 ${getStatusColor(walletData.status).text}`}>
                  {getStatusMessage(walletData.status).title}
                </h3>
                <p className={`text-lg ${getStatusColor(walletData.status).subtext}`}>
                  {getStatusMessage(walletData.status).message}
                </p>
              </div>
            </div>

            {/* Suspicious Activities Summary */}
            <div className="bg-white rounded-xl p-6 mb-6 border border-gray-200">
              <h4 className="font-semibold text-gray-900 mb-4 text-lg">
                Detected Issues ({walletData.activities.length})
              </h4>
              <div className="space-y-3">
                {walletData.activities.map((activity: SuspiciousActivity, index: number) => {
                  // Calculate color based on position - green to red gradient
                  const progress = index / Math.max(walletData.activities.length - 1, 1);
                  let bgColor, textColor, borderColor;
                  
                  if (progress < 0.33) {
                    // Green
                    bgColor = "bg-green-50";
                    textColor = "text-green-900";
                    borderColor = "border-green-300";
                  } else if (progress < 0.66) {
                    // Yellow/Amber
                    bgColor = "bg-amber-50";
                    textColor = "text-amber-900";
                    borderColor = "border-amber-300";
                  } else {
                    // Red
                    bgColor = "bg-red-50";
                    textColor = "text-red-900";
                    borderColor = "border-red-300";
                  }
                  
                  return (
                    <div
                      key={activity.id}
                      className={`flex items-start gap-3 p-3 rounded-lg border ${bgColor} ${textColor} ${borderColor}`}
                    >
                      <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <div className="font-semibold text-sm">{activity.type}</div>
                        <div className="text-sm opacity-90">{activity.description}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Know More Button */}
            <div className="flex justify-center">
              <button
                onClick={() => setIsModalOpen(true)}
                className="px-8 py-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 transition-colors shadow-lg text-lg"
              >
                Know More
              </button>
            </div>
          </div>
        )}

        {/* Features Grid - Only show if not analyzed */}
        {!analyzed && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-xl p-6 shadow-md">
                <div className="w-12 h-12 bg-red-500/20 rounded-lg flex items-center justify-center mb-4">
                  <AlertTriangle className="text-red-400" size={24} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Suspicious Pattern Detection
                </h3>
                <p className="text-gray-300">
                  Identify unusual transaction patterns, high-frequency trading, and potential scam indicators
                </p>
              </div>

              <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-xl p-6 shadow-md">
                <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                  <TrendingUp className="text-blue-400" size={24} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Transaction Analysis
                </h3>
                <p className="text-gray-300">
                  Track transaction volume, frequency, and value patterns over time
                </p>
              </div>

              <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-xl p-6 shadow-md">
                <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center mb-4">
                  <Activity className="text-green-400" size={24} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  Real-time Monitoring
                </h3>
                <p className="text-gray-300">
                  Get instant alerts when suspicious activity is detected on tracked wallets
                </p>
              </div>
            </div>

            {/* Example Addresses */}
            <div className="mt-12 text-center">
              <p className="text-sm text-gray-400 mb-4">Try these example addresses:</p>
              <div className="flex flex-wrap justify-center gap-3">
                <button
                  onClick={() => setAddress("0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb")}
                  className="px-4 py-2 bg-white/10 backdrop-blur-sm border border-white/20 text-white rounded-lg hover:bg-white/20 transition-colors text-sm"
                >
                  0x742d...f0bEb
                </button>
                <button
                  onClick={() => setAddress("0x8894E0a0c962CB723c1976a4421c95949bE2D4E3")}
                  className="px-4 py-2 bg-white/10 backdrop-blur-sm border border-white/20 text-white rounded-lg hover:bg-white/20 transition-colors text-sm"
                >
                  0x8894...2D4E3
                </button>
                <button
                  onClick={() => setAddress("0x1234567890abcdef1234567890abcdef12345678")}
                  className="px-4 py-2 bg-white/10 backdrop-blur-sm border border-white/20 text-white rounded-lg hover:bg-white/20 transition-colors text-sm"
                >
                  0x1234...45678
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Tracker Details Modal */}
      {walletData && (
        <TrackerDetailsModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          walletData={walletData}
        />
      )}
    </div>
  );
}