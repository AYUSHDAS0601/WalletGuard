import { ArrowDownRight, ArrowUpRight, Clock, CheckCircle, XCircle, ExternalLink } from "lucide-react";
import { format } from "date-fns";

interface Transaction {
  id: string;
  hash: string;
  type: "sent" | "received";
  amount: string;
  from: string;
  to: string;
  timestamp: string;
  status: "confirmed" | "pending" | "failed";
}

interface TransactionListProps {
  transactions: Transaction[];
}

export function TransactionList({ transactions }: TransactionListProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "confirmed":
        return <CheckCircle className="text-green-600" size={16} />;
      case "pending":
        return <Clock className="text-amber-600" size={16} />;
      case "failed":
        return <XCircle className="text-red-600" size={16} />;
      default:
        return null;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold">Recent Transactions</h2>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Hash</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">From/To</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {transactions.map((tx) => (
              <tr key={tx.id} className="hover:bg-gray-50 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className={`flex items-center gap-2 ${
                    tx.type === "received" ? "text-green-600" : "text-red-600"
                  }`}>
                    {tx.type === "received" ? (
                      <ArrowDownRight size={18} />
                    ) : (
                      <ArrowUpRight size={18} />
                    )}
                    <span className="font-medium capitalize">{tx.type}</span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-gray-900">{tx.hash}</span>
                    <button className="text-gray-400 hover:text-gray-600">
                      <ExternalLink size={14} />
                    </button>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="font-semibold text-gray-900">{tx.amount} ETH</span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm">
                    <div className="text-gray-600">From: <span className="font-mono">{tx.from}</span></div>
                    <div className="text-gray-600">To: <span className="font-mono">{tx.to}</span></div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                  {format(new Date(tx.timestamp), "MMM d, h:mm a")}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-1">
                    {getStatusIcon(tx.status)}
                    <span className="text-sm capitalize">{tx.status}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
