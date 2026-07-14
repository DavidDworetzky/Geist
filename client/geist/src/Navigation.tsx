import { NavLink } from 'react-router-dom';

function Navigation(props: NavigationProps): JSX.Element {
  const innerElements = props.navigationElements.map((ele, index) => (
    <li key={`${ele.link}-${index}`}>
      <NavLink to={ele.link} className="list-link">
        <span className="nav-icon" aria-hidden="true">
          <svg focusable="false" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
            <path fill="currentColor" d={ele.svg}></path>
          </svg>
        </span>
        <span>{ele.name}</span>
      </NavLink>
    </li>
  ));

  return (
    <nav className="LinkList" aria-label="Navigation">
      <ul>{innerElements}</ul>
    </nav>
  );
}

interface NavigationProps {
  navigationElements: NavigationElement[];
}

interface NavigationElement {
  name: string;
  link: string;
  svg: string;
}

export default Navigation;
