import { useState, useEffect } from "react";
import { useParams, Link } from "react-router";
import { Shield, ArrowLeft, AlertCircle, CheckCircle, Clock, TrendingUp } from "lucide-react";
import { RiskScore } from "../components/RiskScore";
import { SuspiciousActivityList } from "../components/SuspiciousActivityList";
import { TransactionList } from "../components/TransactionList";
import { ActivityChart } from "../components/ActivityChart";
import { ActivityDetailsModal } from "../components/ActivityDetailsModal";

interface WalletData {
  address: string;
  balance: string;
  transactionCount: number;
  firstSeen: string;
  lastActive: string;
  riskScore: number;
  suspiciousActivities: Array<{
    id: string;
    type: string;
    severity: "high" | "medium" | "low";
    description: string;
    timestamp: string;
    details: string;
  }>;
  transactions: Array<{
    id: string;
    hash: string;
    type: "sent" | "received";
    amount: string;
    from: string;
    to: string;
    timestamp: string;
    status: "confirmed" | "pending" | "failed";
  }>;
}

// Mock data generator
const generateMockData = (address: string): WalletData => {
  const riskScore = Math.floor(Math.random() * 100);
  
  const suspiciousActivities = [
    {
      id: "1",
      type: "High-Frequency Trading",
      severity: "high" as const,
      description: "Detected 45 transactions within 10 minutes - possible bot activity",
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      details: "This wallet executed 45 transactions in a 10-minute window, which is characteristic of automated bot trading. The transactions were made to multiple different addresses with varying amounts, suggesting potential wash trading or market manipulation activities.",
    },
    {
      id: "2",
      type: "Large Transfer",
      severity: "medium" as const,
      description: "Transferred 150 ETH to unknown address",
      timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
      details: "A transfer of 150 ETH was made to an address with no prior transaction history with this wallet. The receiving address was created less than 24 hours ago, which is a red flag for potential fund siphoning or preparation for exit scam.",
    },
    {
      id: "3",
      type: "Mixer Interaction",
      severity: "high" as const,
      description: "Interacted with known mixing service",
      timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      details: "The wallet has sent funds to a known cryptocurrency mixer (Tornado Cash). Mixers are used to obscure the origin of funds, which can be legitimate for privacy but is also commonly associated with money laundering activities.",
    },
    {
      id: "4",
      type: "Unusual Pattern",
      severity: "low" as const,
      description: "Transaction timing pattern differs from historical behavior",
      timestamp: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
      details: "Recent transactions occurred between 2 AM - 4 AM local time, which deviates significantly from the wallet's historical activity pattern (typically 9 AM - 6 PM). This could indicate compromised access or automated bot activity.",
    },
  ];

  const transactions = Array.from({ length: 15 }, (_, i) => ({
    id: `tx-${i}`,
    hash: `0x${Math.random().toString(16).slice(2, 18)}...${Math.random().toString(16).slice(2, 10)}`,
    type: Math.random() > 0.5 ? ("sent" as const) : ("received" as const),
    amount: (Math.random() * 10).toFixed(4),
    from: `0x${Math.random().toString(16).slice(2, 10)}...`,
    to: `0x${Math.random().toString(16).slice(2, 10)}...`,
    timestamp: new Date(Date.now() - i * 3 * 60 * 60 * 1000).toISOString(),
    status: "confirmed" as const,
  }));

  return {
    address,
    balance: (Math.random() * 100).toFixed(4),
    transactionCount: 1234,
    firstSeen: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString(),
    lastActive: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    riskScore,
    suspiciousActivities: riskScore > 50 ? suspiciousActivities : suspiciousActivities.slice(0, 1),
    transactions,
  };
};

export function Tracker() {
  const { address } = useParams<{ address: string }>();
  const [loading, setLoading] = useState(true);
  const [walletData, setWalletData] = useState<WalletData | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    // Simulate API call
    setLoading(true);
    setTimeout(() => {
      if (address) {
        setWalletData(generateMockData(address));
      }
      setLoading(false);
    }, 1000);
  }, [address]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Analyzing wallet activity...</p>
        </div>
      </div>
    );
  }

  if (!walletData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">Wallet not found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                to="/"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeft size={20} />
                <span>Back</span>
              </Link>
              <div className="flex items-center gap-2">
                <Shield className="text-blue-600" size={28} />
                <h1 className="text-xl font-bold text-gray-900">WalletGuard</h1>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Wallet Address Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6 border border-gray-200">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <p className="text-sm text-gray-600 mb-1">Tracking Wallet</p>
              <p className="text-lg font-mono font-semibold text-gray-900 break-all">
                {walletData.address}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">Balance</p>
              <p className="text-lg font-semibold text-gray-900">{walletData.balance} ETH</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Transactions</p>
              <p className="text-lg font-semibold text-gray-900">{walletData.transactionCount}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">First Seen</p>
              <p className="text-lg font-semibold text-gray-900">
                {new Date(walletData.firstSeen).toLocaleDateString()}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Last Active</p>
              <p className="text-lg font-semibold text-gray-900">
                {new Date(walletData.lastActive).toLocaleTimeString()}
              </p>
            </div>
          </div>
        </div>

        {/* Risk Score */}
        <div className="mb-6">
          <RiskScore score={walletData.riskScore} />
        </div>

        {/* Suspicious Activities Alert */}
        {walletData.suspiciousActivities.length > 0 && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start justify-between">
            <div className="flex items-start gap-3">
              <AlertCircle className="text-red-600 flex-shrink-0 mt-0.5" size={20} />
              <div>
                <p className="font-semibold text-red-900">
                  {walletData.suspiciousActivities.length} suspicious {walletData.suspiciousActivities.length === 1 ? 'activity' : 'activities'} detected
                </p>
                <p className="text-sm text-red-700">
                  This wallet has triggered multiple security alerts. Exercise caution.
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsModalOpen(true)}
              className="ml-4 px-6 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap"
            >
              Know More
            </button>
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Suspicious Activities */}
          <SuspiciousActivityList activities={walletData.suspiciousActivities} />

          {/* Activity Chart */}
          <ActivityChart />
        </div>

        {/* Transaction List */}
        <TransactionList transactions={walletData.transactions} />
      </div>

      {/* Activity Details Modal */}
      <ActivityDetailsModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        activities={walletData.suspiciousActivities}
        address={walletData.address}
      />
    </div>
  );
}