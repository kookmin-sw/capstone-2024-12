import { useNavigate } from 'react-router-dom';
import { useEffect } from 'react';

export default function Index(props) {
  const navigate = useNavigate();
  useEffect(() => {
    navigate('/dashboard');
  }, []);
  return <></>;
}
