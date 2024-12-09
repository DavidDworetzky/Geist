import React from 'react';
import logo from './logo.svg';
import geist_avatar from './geist_avatar.png'
import './App.css';
import Navigation from './Navigation';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './Home';
import Chat from './Chat';


function App() {
  const links = [{
    name: 'Home',
    link: '/',
    svg: 'M216 0h80c13.3 0 24 10.7 24 24v168h87.7c17.8 0 26.7 21.5 14.1 34.1L269.7 378.3c-7.5 7.5-19.8 7.5-27.3 0L90.1 226.1c-12.6-12.6-3.7-34.1 14.1-34.1H192V24c0-13.3 10.7-24 24-24zm296 376v112c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V376c0-13.3 10.7-24 24-24h146.7l49 49c20.1 20.1 52.5 20.1 72.6 0l49-49H488c13.3 0 24 10.7 24 24zm-124 88c0-11-9-20-20-20s-20 9-20 20 9 20 20 20 20-9 20-20zm64 0c0-11-9-20-20-20s-20 9-20 20 9 20 20 20 20-9 20-20z'
  },
  {
    name: 'chat',
    link: '/chat',
    svg: 'M336.5 160C322 70.7 287.8 8 248 8s-74 62.7-88.5 152h177zM152 256c0 22.2 1.2 43.5 3.3 64h185.3c2.1-20.5 3.3-41.8 3.3-64s-1.2-43.5-3.3-64H155.3c-2.1 20.5-3.3 41.8-3.3 64zm324.7-96c-28.6-67.9-86.5-120.4-158-141.6 24.4 33.8 41.2 84.7 50 141.6h108zM177.2 18.4C105.8 39.6 47.8 92.1 19.3 160h108c8.7-56.9 25.5-107.8 49.9-141.6zM487.4 192H372.7c2.1 21 3.3 42.5 3.3 64s-1.2 43-3.3 64h114.6c5.5-20.5 8.6-41.8 8.6-64s-3.1-43.5-8.5-64zM120 256c0-21.5 1.2-43 3.3-64H8.6C3.2 212.5 0 233.8 0 256s3.2 43.5 8.6 64h114.6c-2-21-3.2-42.5-3.2-64zm39.5 96c14.5 89.3 48.7 152 88.5 152s74-62.7 88.5-152h-177zm159.3 141.6c71.4-21.2 129.4-73.7 158-141.6h-108c-8.8 56.9-25.6 107.8-50 141.6zM19.3 352c28.6 67.9 86.5 120.4 158 141.6-24.4-33.8-41.2-84.7-50-141.6h-108z'
  }]
  return (

    <div id="Container" className="Wrapper">
      <aside className="App-header">
      <Navigation navigationElements={links}/>
        <img src={geist_avatar} className="App-logo" alt="logo" />
        <p>
          Welcome to Geist!, an open source LLM Workbench!
        </p>
        <a
          className="App-link"
          href="https://github.com/DavidDworetzky/Fastcasso"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn more
        </a>
      </aside>
      <main className="Content">
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Chat/>}/>
            <Route path="/chat" element={<Chat/>}/>
            <Route path="/chat/:chatId" element={<Chat/>}/>
          </Routes>
        </BrowserRouter>

      </main>
    </div>
  );
}

export default App;
