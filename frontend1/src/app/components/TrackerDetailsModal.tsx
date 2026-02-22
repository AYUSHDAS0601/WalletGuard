import { X } from "lucide-react";
import { RiskScore } from "./RiskScore";
import { SuspiciousActivityList } from "./SuspiciousActivityList";
import { TransactionList } from "./TransactionList";
import { ActivityChart } from "./ActivityChart";

interface TrackerDetailsModalProps {
  isOpen: boolean;
  onClose: () => void;
  walletData: any;
}

// Generate mock transactions
const generateMockTransactions = () => {
  return Array.from({ length: 15 }, (_, i) => ({
    id: `tx-${i}`,
    hash: `0x${Math.random().toString(16).slice(2, 18)}...${Math.random().toString(16).slice(2, 10)}`,
    type: Math.random() > 0.5 ? ("sent" as const) : ("received" as const),
    amount: (Math.random() * 10).toFixed(4),
    from: `0x${Math.random().toString(16).slice(2, 10)}...`,
    to: `0x${Math.random().toString(16).slice(2, 10)}...`,
    timestamp: new Date(Date.now() - i * 3 * 60 * 60 * 1000).toISOString(),
    status: "confirmed" as const,
  }));
};

export function TrackerDetailsModal({ isOpen, onClose, walletData }: TrackerDetailsModalProps) {
  if (!isOpen) return null;

  const transactions = generateMockTransactions();

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-60 transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="flex min-h-screen items-start justify-center p-4 pt-8">
        <div className="relative bg-gray-50 rounded-2xl shadow-2xl max-w-7xl w-full max-h-[90vh] overflow-hidden">
          {/* Header */}
          <div className="sticky top-0 bg-white border-b border-gray-200 px-8 py-6 flex items-center justify-between z-10 shadow-sm">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Wallet Analysis Details</h2>
              <p className="text-sm text-gray-600 font-mono mt-1">{walletData.address}</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X size={24} className="text-gray-600" />
            </button>
          </div>

          {/* Content */}
          <div className="px-8 py-6 overflow-y-auto max-h-[calc(90vh-100px)]">
            {/* Wallet Info */}
            <div className="bg-white rounded-lg shadow-sm p-6 mb-6 border border-gray-200">
              <h3 className="font-semibold text-gray-900 mb-4">Wallet Information</h3>
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

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* Suspicious Activities */}
              <SuspiciousActivityList activities={walletData.activities} />

              {/* Activity Chart */}
              <ActivityChart />
            </div>

            {/* Transaction List */}
            <TransactionList transactions={transactions} />
          </div>

          {/* Footer */}
          <div className="sticky bottom-0 bg-white border-t border-gray-200 px-8 py-4 flex justify-end shadow-sm">
            <button
              onClick={onClose}
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
