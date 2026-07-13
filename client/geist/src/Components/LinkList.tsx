import { NavLink } from 'react-router-dom';

function LinkList(props: LinkListProps): JSX.Element {
  const innerElements = props.listItems.map((ele, index) => {
    const isLast = props.listItems.length === index + 1;
    return (
      <li ref={isLast ? props.lastItemRef : null} key={index}>
        <NavLink to={ele.link} className="list-link" onClick={() => props.onItemSelect?.(ele)}>
          <span className="chat-history-item">{ele.name}</span>
        </NavLink>
      </li>
    );
  });

  return (
    <div className="LinkList">
      <ul className="relative">
        {innerElements}
      </ul>
    </div>
  );
}

export interface LinkListProps {
  listItems: ListItem[];
  lastItemRef?: (node: any) => void;
  onItemSelect?: (item: ListItem) => void;
}

export interface ListItem {
  name: string;
  link: string;
  date: Date;
}

export default LinkList;
