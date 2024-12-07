function LinkList(props: LinkListProps): JSX.Element {
    const innerElements = props.listItems.map((ele, index) => {
        return (
            <li key={index} className="relative nav-item">
                <a href={ele.link} className="flex items-center text-lg py-4 px-12 h-20 overflow-hidden text-black-700 text-ellipsis whitespace-nowrap rounded hover:text-gray-900 hover:bg-gray-100 transition duration-300 ease-in-out" data-mdb-ripple="true" data-mdb-ripple-color="dark">
                    <span>{ele.name}</span>
                </a>
            </li>
        )
    })
    return (
        <div>
            <ul className="relative">
                {innerElements}
            </ul>
        </div>)
}

interface LinkListProps {
    listItems: ListItem[];
}

interface ListItem {
    name: string;
    link: string;
}

export default LinkList
