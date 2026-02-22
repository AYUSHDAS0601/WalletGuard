import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const mockData = [
  { date: "Jan 15", transactions: 12 },
  { date: "Jan 16", transactions: 19 },
  { date: "Jan 17", transactions: 8 },
  { date: "Jan 18", transactions: 25 },
  { date: "Jan 19", transactions: 45 },
  { date: "Jan 20", transactions: 32 },
  { date: "Today", transactions: 28 },
];

export function ActivityChart() {
  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
      <h2 className="text-xl font-semibold mb-4">Transaction Activity (Last 7 Days)</h2>
      
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={mockData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="date" 
            tick={{ fill: '#6b7280', fontSize: 12 }}
            stroke="#9ca3af"
          />
          <YAxis 
            tick={{ fill: '#6b7280', fontSize: 12 }}
            stroke="#9ca3af"
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
            }}
            cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
          />
          <Bar 
            dataKey="transactions" 
            fill="#3b82f6" 
            radius={[8, 8, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Peak Activity</span>
          <span className="font-semibold text-gray-900">45 transactions on Jan 19</span>
        </div>
      </div>
    </div>
  );
}
