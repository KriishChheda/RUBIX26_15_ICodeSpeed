import { createContext, useState, useCallback, useRef } from 'react';
import Notification from '../components/Notification';

export const NotificationContext = createContext();

let notificationId = 0;

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);
  const activeTypesRef = useRef(new Map()); // Track active notification types with their IDs

  const addNotification = useCallback((message, type = 'info', duration = 3000, category = null) => {
    // Create a unique key for this notification type
    // Use category if provided (for proctoring alerts), otherwise use message
    const typeKey = category || message;

    // Check if this type of notification is already active
    if (activeTypesRef.current.has(typeKey)) {
      console.log(`[Notification] Blocked duplicate notification of type: ${typeKey}`);
      return activeTypesRef.current.get(typeKey); // Return existing notification ID
    }

    const id = ++notificationId;
    const newNotification = { id, message, type, duration, category: typeKey };
    
    // Track this notification type as active
    activeTypesRef.current.set(typeKey, id);
    
    setNotifications(prev => [...prev, newNotification]);
    
    return id;
  }, []);

  const removeNotification = useCallback((id) => {
    setNotifications(prev => {
      // Find the notification being removed
      const notification = prev.find(n => n.id === id);
      
      // Remove it from active types tracking
      if (notification && notification.category) {
        activeTypesRef.current.delete(notification.category);
      }
      
      return prev.filter(notification => notification.id !== id);
    });
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
    activeTypesRef.current.clear();
  }, []);

  const notify = {
    success: (message, duration, category) => addNotification(message, 'success', duration, category),
    error: (message, duration, category) => addNotification(message, 'error', duration, category),
    warning: (message, duration, category) => addNotification(message, 'warning', duration, category),
    info: (message, duration, category) => addNotification(message, 'info', duration, category),
  };

  return (
    <NotificationContext.Provider value={{ addNotification, removeNotification, clearAll, notify }}>
      {children}
      
      {/* Notification Container - Bottom Right */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col-reverse gap-3 pointer-events-none">
        {notifications.map((notification) => (
          <div key={notification.id} className="pointer-events-auto">
            <Notification
              id={notification.id}
              message={notification.message}
              type={notification.type}
              duration={notification.duration}
              onClose={removeNotification}
            />
          </div>
        ))}
      </div>
    </NotificationContext.Provider>
  );
};
